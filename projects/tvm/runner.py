"""
projects/tvm/runner.py — one FusionFuzz execution of a TVMScript seed.

    python3 runner.py PATH.py --mode parse|build|run|diff [--target T]
                      [--opt-level N] [--seed S] [--num-passes K]
                      [--tir-pipeline default|tirx]

Runs inside the ffe-tvm container with PYTHONPATH pointing at the built
TVM. The seed defines `Module` (an IRModule) through TVMScript; this
script is the part of the oracle that needs TVM itself:

  parse   TVMScript parsing + the IRModule's structural verification.
  build   `--num-passes` passes drawn (deterministically from --seed)
          from the no-argument TIR/Relax transforms, then
          `tvm.compile(mod, target)`.
  run     build, then execute the module's entry (a PrimFunc whose
          parameters are all static buffers, or a Relax `main` with static
          tensors) on random inputs — the program's own runtime checks
          (bounds asserts in the generated code, `T.assert`) are the oracle.
  diff    run at the requested target/opt level and at the baseline
          `llvm -opt-level=0`; outputs that differ beyond tolerance are
          a miscompilation candidate.

Markers on stdout tell projects/tvm/analyzer.py what happened without
parsing Python tracebacks: FFL_REJECTED (the seed is not a valid program:
TVMScript error, verification failure, unsupported op, a Python error
inside the seed), FFL_INTERNAL_ERROR (ICHECK / InternalError: the
compiler's own invariant), FFL_MISMATCH, FFL_OK. Native crashes (segfault,
LLVM assertion) kill the process before any marker.
"""

import argparse
import os
import random
import shutil
import sys
import traceback

import numpy as np


import re
# passes that only make sense at their point of the lowering pipeline
# (they consume a target context, host/device split, packed API, thread
# bindings...) and ICHECK their preconditions on anything else
_PIPELINE_ONLY_RE = re.compile(r"Lower|MakePacked|SplitHost|Thread|Warp|Device|Vectorize|Storage|"
                               r"Flatten|Inject|Bind|Unroll|Narrow|Hoist|Compact|Annotate|Coproc|"
                               r"Combine|Merge|Legalize|Attach|Realize|Lift|Manifest|Convert|"
                               r"Extract|Default|Verify|Apply|Instrument|Profile|Debug|Random|"
                               r"Texture|Async|Pipeline|Ptx|Cuda|Vulkan|Metal|OpenCL|Hexagon|Fp8|Fp16")
#: The target refusing the module, not an invariant of its own: a device
#: capability the drawn target does not declare (Vulkan's Int8/Int64/
#: Float16 capabilities, a shared-memory or thread limit), a toolchain the
#: image does not have (ROCm device bitcode), or a codegen limit stated in
#: the message. All of these are rejections of the input for that target.
_TARGET_PRECONDITION_RE = re.compile(
    r"does not support \w+ capability|please either add -support|"
    r"could not find bitcode|Cannot find (?:CUDA|ROCm) path|"
    r"cannot be larger than|exceeds? the (?:maximum|limit)|"
    r"does not support (?:the )?(?:dtype|data type|type)|"
    r"not supported (?:by|on|for) (?:this )?target|no longer supported|"
    r"Unsupported (?:dtype|data type|type|target|device)|"
    # structural: a GPU codegen needs kernels (thread bindings) and only
    # allows shared/local allocations inside one. A module that has none
    # is not a program for that target.
    r"Can only allocate shared or local memory inside kernel|"
    r"thread (?:extent|binding).*(?:not|missing)|"
    r"must be called after|expects? to be run after|"
    r"cannot be scheduled|no thread (?:axis|binding)|"
    # the tirx pipeline saying it cannot lower this module for this target
    # (it prints the whole program after the message)
    r"Failed to lower the TIRx program|"
    # the module itself is ill-formed for any backend: two exported
    # PrimFuncs with the same global_symbol
    r"Duplicate PrimFunc global_symbol|"
    # a backend stating a limit of its allocation model
    r"requires a finite compile-time upper bound|WebGPU allocation|"
    # "blockIdx.z is not supported in WebGPU": a backend limit
    r"is not supported in \w+|"
    # "CodeGenWebGPU only allows constant thread group size"
    r"only allows |"
    # a codegen declining a construct or a type it cannot express
    r"not implemented|Cannot convert type|"
    # the module carries undefined variables (the well-formedness
    # complaint a later pass makes), or a Relax operator that a legalise
    # pass was supposed to lower first
    r"undefined\.size\(\) == 0|cannot emit this Relax operator directly|"
    # a codegen stating a type or memory scope it does not implement
    r"do not support|only supports? \w+|Cannot allocate global memory|"
    r"Device kernel may only contain", re.I)

_PRECONDITION_RE = re.compile(r"Require the|must be called|context required|Please set|"
                              r"should be run|expected to be run|only supports|not supported|"
                              r"is not supported|already exists|Unknown device id|requires? .*pass",
                              re.I)


def _load_module(path):
    # TVMScript's parser reads the class source through inspect, so the
    # seed has to be a real module registered in sys.modules with a file.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ffl_seed", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ffl_seed"] = module
    spec.loader.exec_module(module)   # the seed is the input under test
    ns = vars(module)
    import tvm
    mods = [v for v in ns.values() if isinstance(v, tvm.IRModule)]
    if "Module" in ns and isinstance(ns["Module"], tvm.IRModule):
        return ns["Module"]
    if mods:
        return mods[0]
    funcs = [v for v in ns.values() if isinstance(v, (tvm.tirx.PrimFunc,))] if hasattr(tvm, "tirx") else []
    if funcs:
        return tvm.IRModule.from_expr(funcs[0])
    raise RuntimeError("seed defines no IRModule")


def _pass_pool():
    """No-argument pass constructors of the TIR and Relax transform
    namespaces, discovered at run time so the pool follows the checkout."""
    import tvm
    pool = []
    for modname in ("tirx.transform", "s_tir.transform", "tir.transform", "relax.transform"):
        obj = tvm
        try:
            for part in modname.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        for name in sorted(dir(obj)):
            if not name[0].isupper() or _PIPELINE_ONLY_RE.search(name):
                continue
            ctor = getattr(obj, name)
            if not callable(ctor):
                continue
            try:
                p = ctor()
            except Exception:
                continue
            if isinstance(p, tvm.transform.Pass):
                pool.append((f"{modname}.{name}", p))
    return pool


def _is_static(shape):
    return all(isinstance(int(d) if hasattr(d, "value") or isinstance(d, int) else d, int) for d in shape) if shape else True


def _has_thread_binding(mod):
    """True when some function already binds a thread axis — printing the
    module is enough to tell, and is version-independent."""
    try:
        text = mod.script()
    except Exception:
        return False
    return ("thread_binding" in text or "launch_thread" in text
            or "env_thread" in text or "threadIdx" in text)


def _entry(mod):
    """(kind, name, params) of a runnable entry: a Relax `main` or a
    PrimFunc whose parameters are all static tensors/buffers. In this TVM
    a parameter Var carries `shape`/`dtype` directly (TIR buffers and
    Relax tensors alike); older IR exposes `struct_info` or a
    `buffer_map` entry instead — all three are accepted."""
    import tvm
    from tvm import relax

    def _typed(f, p):
        si = getattr(p, "struct_info", None)
        if si is not None and isinstance(si, relax.TensorStructInfo) and si.shape is not None:
            return [int(d) for d in si.shape.values], str(si.dtype)
        buf = p if hasattr(p, "shape") and hasattr(p, "dtype") else \
            (f.buffer_map.get(p) if hasattr(f, "buffer_map") else None)
        if buf is None:
            return None
        return [int(d) for d in buf.shape], str(buf.dtype)

    funcs = list(mod.functions.items())
    funcs.sort(key=lambda x: (not isinstance(x[1], relax.Function), x[0].name_hint != "main"))
    for gv, f in funcs:
        kind = "relax" if isinstance(f, relax.Function) else "prim"
        if kind == "relax" and gv.name_hint != "main":
            continue
        params = []
        try:
            for p in f.params:
                t = _typed(f, p)
                if t is None:
                    params = None; break
                params.append((str(getattr(p, "name_hint", None) or getattr(p, "name", None) or p), t[0], t[1]))
        except Exception:
            params = None
        if params:
            return (kind, gv.name_hint, params)
    return None


_BUF_DECL_RE = re.compile(r"\b(\w+)\s*(?::\s*T\.Buffer\(\s*\(?([\d,\s]+?)\)?\s*,|=\s*T\.(?:match_buffer|alloc_buffer|sblock_alloc_buffer)\(\s*(?:\w+\s*,\s*)?[\[(]([\d,\s]+?)[\])])")
_CONST_ACCESS_RE = re.compile(r"\b(\w+)\[([\d,\s]+)\]")


_GRID_RE = re.compile(r"for\s+([\w,\s]+?)\s+in\s+T\.grid\(([\d,\s]+)\)")
_RANGE_RE = re.compile(r"for\s+(\w+)\s+in\s+(?:range|T\.serial|T\.parallel|T\.vectorized|T\.unroll)\((?:0\s*,\s*)?(\d+)\)")
_REMAP_RE = re.compile(r"([\w,\s]+?)\s*=\s*T\.axis\.remap\(\s*\"[SR]+\"\s*,\s*\[([\w,\s]+)\]")
_AXIS_RE = re.compile(r"(\w+)\s*=\s*T\.axis\.(?:S|R|spatial|reduce)\(\s*(\d+)\s*,\s*(\w+)")
_VAR_ACCESS_RE = re.compile(r"\b(\w+)\[([^\]]+)\]")
_INDEX_TERM_RE = re.compile(r"^\s*(\w+)\s*(?:([+-])\s*(\d+))?\s*$")


def _loop_oob(script, shapes):
    """`A[vi, vj + 2]` where `vj` spans the whole dimension: the last
    iterations read past the buffer. Loop extents come from `T.grid` /
    `range` / `T.serial`, axis extents from `T.axis.remap` (which inherits
    the loop's) or `T.axis.S(N, i)`. Returns a message or None."""
    extent = {}
    for m in _GRID_RE.finditer(script):
        names = [n.strip() for n in m.group(1).split(",") if n.strip()]
        sizes = [int(x) for x in m.group(2).replace(" ", "").split(",") if x]
        for n, sz in zip(names, sizes):
            extent[n] = sz
    for m in _RANGE_RE.finditer(script):
        extent[m.group(1)] = int(m.group(2))
    for m in _REMAP_RE.finditer(script):
        outs = [n.strip() for n in m.group(1).split(",") if n.strip()]
        ins = [n.strip() for n in m.group(2).split(",") if n.strip()]
        for o, i in zip(outs, ins):
            if i in extent:
                extent[o] = extent[i]
    for m in _AXIS_RE.finditer(script):
        extent[m.group(1)] = int(m.group(2))
    if not extent:
        return None
    for m in _VAR_ACCESS_RE.finditer(script):
        name, idx = m.group(1), m.group(2)
        if name not in shapes:
            continue
        terms = idx.split(",")
        if len(terms) != len(shapes[name]):
            continue
        for term, dim in zip(terms, shapes[name]):
            t = _INDEX_TERM_RE.match(term)
            if not t or t.group(1) not in extent:
                continue
            off = int(t.group(3) or 0) * (1 if (t.group(2) or "+") == "+" else -1)
            lo, hi = off, extent[t.group(1)] - 1 + off
            if lo < 0 or hi >= dim:
                return f"{name}[{idx.strip()}] with {t.group(1)} in [0, {extent[t.group(1)]}) outside dimension {dim}"
    return None


def _constant_oob(script):
    """A buffer access with all-constant indices outside a buffer with an
    all-constant shape, e.g. `B[0, 0] = A[2, 2]` with `A: T.Buffer((2, 3))`.

    TVM's `tirx.instrument_bound_checkers` is accepted but does nothing in
    this checkout (InstrumentBoundCheckers only instruments accesses carrying
    `buffer_bound` attributes, which no pass emits any more), so an
    out-of-bounds read runs and returns whatever is next in memory. That
    made `bad_load`-style seeds report "nan vs 0.0 (core-avx2 vs llvm)" as
    a miscompile. Such a program is undefined, not a test of the compiler,
    and is rejected before it is executed. Returns a message or None."""
    shapes = {}
    for m in _BUF_DECL_RE.finditer(script):
        dims = m.group(2) or m.group(3)
        try:
            shapes[m.group(1)] = [int(d) for d in dims.replace(" ", "").split(",") if d]
        except ValueError:
            continue
    for m in _CONST_ACCESS_RE.finditer(script):
        name, idx = m.group(1), m.group(2)
        if name not in shapes:
            continue
        try:
            indices = [int(d) for d in idx.replace(" ", "").split(",") if d]
        except ValueError:
            continue
        shape = shapes[name]
        if len(indices) != len(shape):
            continue
        for i, (v, n) in enumerate(zip(indices, shape)):
            if v < 0 or v >= n:
                return f"{name}[{idx}] outside shape {shape}"
    return _loop_oob(script, shapes)


_ALLOC_RE = re.compile(r"^\s*(\w+)\s*=\s*T\.(?:alloc_buffer|sblock_alloc_buffer)\(", re.M)


def _read_before_write(script):
    """The first access to a buffer the function allocates itself is a read.

    TIR executes in program order, so an internal buffer whose first
    textual access is a read is read before anything wrote it: a reduction
    without `T.init()`, or an opaque block that *declares* a write through
    `access_ptr("w")` and performs none (`T.evaluate` of an address is not
    a store). The value read is whatever the allocation held, which differs
    between builds — "1.0 vs 3.0 (core-avx2 opt 2 vs llvm opt 0)" was
    `garbage * 2 + 1` — so the program cannot be used as a differential
    test. Returns the buffer name or None. Same-line `A[i] = A[i-1] + 1`
    counts as a write, so the check is lenient inside loops."""
    names = set(_ALLOC_RE.findall(script))
    if not names:
        return None
    seen = set()
    for line in script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "T.reads(" in stripped or "T.writes(" in stripped:
            continue
        m = re.match(r"(\w+)\[[^\]]*\]\s*=(?!=)", stripped)
        written = m.group(1) if m else None
        for name in names:
            if name in seen:
                continue
            if re.search(rf"\b{re.escape(name)}\[", stripped):
                seen.add(name)
                if written != name:
                    return name
    return None


def _random_inputs(params, rng):
    import tvm
    arrays = []
    for _n, shape, dtype in params:
        if dtype.startswith("float") or dtype.startswith("bfloat"):
            a = rng.standard_normal(size=shape).astype("float32")
            a = a.astype(dtype if dtype in ("float32", "float64", "float16") else "float32")
        elif dtype.startswith("int") or dtype.startswith("uint"):
            a = rng.integers(0, 8, size=shape).astype(dtype)
        elif dtype == "bool":
            a = rng.integers(0, 2, size=shape).astype("bool")
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        arrays.append(tvm.runtime.tensor(a) if hasattr(tvm.runtime, "tensor") else tvm.nd.array(a))
    return arrays


#: Target kinds whose codegen emits device code. All of them are
#: compile-only here: no GPU is present, and nothing is ever launched.
#: `cuda`, `opencl`, `metal` and `webgpu` are source-level codegens (CUDA
#: C++, OpenCL C, Metal, WGSL) and need no toolkit at all; `nvptx` and
#: `rocm` go through LLVM's NVPTX/AMDGPU back ends and need libdevice /
#: the ROCm device bitcode; `vulkan` needs a USE_VULKAN build (SPIR-V).
GPU_KINDS = ("cuda", "nvptx", "rocm", "vulkan", "opencl", "metal", "webgpu")

#: Integer target options that are device limits rather than ISA
#: selection. Drawn by the driver as `-max_num_threads=...` etc.
_INT_TARGET_OPTS = (
    "max_num_threads", "max_threads_per_block", "max_shared_memory_per_block",
    "thread_warp_size", "registers_per_block", "l2_cache_size_bytes",
    "max_function_args", "max_block_size_x", "max_block_size_y", "max_block_size_z",
    "max_push_constants_size", "max_uniform_buffer_range", "max_storage_buffer_range",
    "max_per_stage_descriptor_storage_buffer", "supported_subgroup_operations",
    "texture_spatial_limit", "texture_depth_limit", "image_base_address_alignment",
    "vulkan_api_version", "max_spirv_version", "driver_version",
)
_BOOL_TARGET_OPTS = tuple(
    ["supports_float16", "supports_float32", "supports_float64", "supports_int8",
     "supports_int16", "supports_int32", "supports_int64", "supports_8bit_buffer",
     "supports_16bit_buffer", "supports_storage_buffer_storage_class",
     "supports_push_descriptor", "supports_dedicated_allocation",
     "supports_integer_dot_product", "supports_cooperative_matrix",
     "supports_subgroups"])


#: Below this magnitude a float is subnormal or as good as zero, and two
#: back ends may legitimately disagree: -mcpu=native code can flush
#: subnormals to zero where the -O0 baseline keeps them. Reporting those as
#: miscompilations produced three "mismatches" in one batch whose values
#: were 1.4e-45 and -0.0.
_FLOAT_NOISE_FLOOR = 1e-30


def _float_disagreement(a, b):
    """(index, a_value, b_value) of the first element where two float
    results really disagree, or None.

    numpy's allclose gives one boolean for the whole array, and the index
    of the largest absolute difference is not necessarily an element that
    failed the test — with NaNs in the output it reported "nan vs nan",
    which says nothing. This compares elementwise, treats two NaNs and two
    infinities of the same sign as equal, ignores pairs that are both below
    the noise floor, and names an element that actually differs.
    """
    af = a.astype("float64", copy=False)
    bf = b.astype("float64", copy=False)
    both_nan = np.isnan(af) & np.isnan(bf)
    same_inf = np.isinf(af) & np.isinf(bf) & (np.sign(af) == np.sign(bf))
    tiny = (np.abs(af) < _FLOAT_NOISE_FLOOR) & (np.abs(bf) < _FLOAT_NOISE_FLOOR)
    with np.errstate(invalid="ignore"):
        close = np.isclose(af, bf, rtol=1e-3, atol=1e-4, equal_nan=True)
    bad = ~(close | both_nan | same_inf | tiny)
    if not bad.any():
        return None
    idx = np.unravel_index(int(np.argmax(bad)), a.shape)
    return idx, a[idx], b[idx]


def _make_target(spec):
    """`llvm -mcpu=x -opt-level=N -mtriple=t -mattr=+a,+b`, or a GPU kind
    with its own options (`cuda -arch=sm_90 -max_num_threads=512`) — the
    form the driver draws, TVM's old CLI syntax — as a Target object; the
    CLI string form is no longer accepted by tvm.target.Target."""
    import tvm
    if not isinstance(spec, str):
        return spec
    parts = spec.split()
    kind = parts[0]
    d = {"kind": kind}
    for p in parts[1:]:
        if not p.startswith("-") or "=" not in p:
            continue
        k, v = p[1:].split("=", 1)
        if k == "opt-level":
            continue          # not a target option in this TVM
        elif k == "mattr":
            d[k] = v.split(",")
        elif k in _INT_TARGET_OPTS:
            d[k] = int(v)
        elif k in _BOOL_TARGET_OPTS:
            d[k] = v.lower() in ("1", "true", "on")
        else:
            d[k] = v
    return tvm.target.Target(d)


def _device_sources(lib):
    """(kind, source) for each device module the codegen produced. On a
    USE_CUDA=OFF/USE_VULKAN=OFF build these come back from the fallback
    modules, which hold the generated source instead of a loaded binary —
    exactly what a compile-only run wants."""
    out = []
    try:
        mods = lib.mod.imports
    except Exception:
        return out
    for m in mods:
        try:
            out.append((m.kind, m.inspect_source("")))
        except Exception:
            continue
    return out


_EMPTY_KERNEL_RE = re.compile(r"__global__|kernel void|\bvoid\s+\w+_kernel|OpEntryPoint|@compute")


def _check_device_source(kinds_and_sources, target_kind):
    """The device-codegen oracle: the target's codegen must have produced
    device source, and it must not be empty or truncated. A module that
    compiles to nothing for a GPU target has lost its kernel, which the
    CPU path cannot show."""
    if not kinds_and_sources:
        return "no device module produced"
    for kind, src in kinds_and_sources:
        if not src or not src.strip():
            return f"{kind} device module has empty source"
    return None


def _cross_compile_cuda(src, arch, tools):
    """Compile generated CUDA C++ with nvcc (compile-only, --ptx). Returns
    (rc, output). The toolkit is in the image; no GPU is involved."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cu = os.path.join(d, "kernel.cu")
        with open(cu, "w") as f:
            f.write(src)
        cmd = [tools, f"-arch={arch}", "--ptx", "-o", os.path.join(d, "kernel.ptx"),
               "-w", "-Xcicc", "-O3", cu]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        return p.returncode, (p.stdout or "") + (p.stderr or "")


def _compiles_without_passes(mod, target, args, kw):
    """Is an InternalError seen *with* the drawn passes attributable to
    them rather than to the compiler on this module?

    True when the module compiles for the same target with no drawn
    passes, and also when it is *rejected* without them (a target
    precondition, a diagnostic): in both cases the target never accepted
    this module on its own terms, so what the passes did to it is not a
    finding. False only when the compiler fails an internal check on the
    unmodified module too — that is the genuine case."""
    import tvm
    try:
        with tvm.transform.PassContext(opt_level=args.opt_level):
            tvm.compile(mod, target=_make_target(target), **kw)
        return True
    except tvm.error.InternalError as e:
        return bool(_PRECONDITION_RE.search(str(e)) or _TARGET_PRECONDITION_RE.search(str(e)))
    except Exception:
        return True


def _run_once(mod, target, entry, inputs):
    import tvm
    from tvm import relax
    lib = tvm.compile(mod, target=_make_target(target))
    kind, name, _params = entry
    if kind == "prim":
        rt = lib.jit() if hasattr(lib, "jit") else lib
        f = rt[name]
        f(*inputs)
        return [x.numpy() for x in inputs]
    vm = relax.VirtualMachine(lib, tvm.cpu())
    out = vm["main"](*inputs)
    outs = out if isinstance(out, (list, tuple)) else [out]
    res = []
    for o in outs:
        try:
            res.append(o.numpy())
        except Exception:
            res.append(np.array([0]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--mode", default="build")
    ap.add_argument("--target", default="llvm")
    ap.add_argument("--opt-level", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-passes", type=int, default=0)
    ap.add_argument("--tir-pipeline", default="default")
    ap.add_argument("--nvcc-arch", default="",
                    help="compile the generated CUDA source with nvcc for this arch")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import tvm
    try:
        mod = _load_module(args.path)
    except tvm.error.InternalError as e:
        # The TVMScript parser reports several malformed-input conditions
        # through ICHECK (duplicate function, too many indices, undefined
        # symbol binding) — robustness issues, but the input is invalid, so
        # the run is a rejection; the message is kept for the record.
        print(f"FFL_REJECTED parse-icheck: {str(e)[:200]}"); sys.exit(1)
    except Exception as e:  # TVMScript diagnostics, seed-level Python errors
        print(f"FFL_REJECTED parse: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)

    # TVM's own well-formedness verifiers: a module they reject is not a
    # valid program, and an ICHECK a later pass raises on it is the pass
    # assuming what the verifier guarantees, not a defect.
    checks = []
    for modname, fn in (("tirx.analysis", "verify_well_formed"), ("s_tir.analysis", "verify_well_formed"),
                        ("relax.analysis", "well_formed")):
        try:
            obj = tvm
            for part in modname.split("."):
                obj = getattr(obj, part)
            checks.append((modname, getattr(obj, fn)))
        except AttributeError:
            pass
    for modname, fn in checks:
        try:
            ok = fn(mod)
        except Exception as e:
            msg = str(e)
            # the tirx verifier declines s_tir blocks (and vice versa):
            # not a verdict on the module
            if "does not support" in msg:
                continue
            # the verifiers report through InternalError; that is their
            # diagnostic, and the module is not well-formed
            print(f"FFL_REJECTED not well-formed ({modname}): {type(e).__name__}: {msg[-200:]}"); sys.exit(1)
        if ok is False:
            print(f"FFL_REJECTED not well-formed ({modname})"); sys.exit(1)

    if args.mode == "parse":
        print("FFL_OK parse"); return

    target = args.target      # opt level goes through PassContext only
    tgt_obj = _make_target(target)
    tgt_kind = tgt_obj.kind.name if hasattr(tgt_obj.kind, "name") else str(tgt_obj.kind)
    if tgt_kind in GPU_KINDS:
        # A GPU codegen only accepts kernels: loops bound to thread axes,
        # allocations inside one. Most of the corpus is written without
        # bindings (it targets llvm), and compiling it for a GPU target
        # would just be refused. TVM's own DefaultGPUSchedule binds the
        # remaining loops, which is what its GPU tests do; a module that
        # already has bindings passes through. Failures here are the pass
        # declining the module, so the run is a rejection.
        try:
            sched = tvm.s_tir.transform.DefaultGPUSchedule()
        except AttributeError:
            sched = None
        if sched is not None and not _has_thread_binding(mod):
            try:
                with tgt_obj:
                    mod = sched(mod)
            except tvm.error.InternalError as e:
                print(f"FFL_REJECTED gpu-schedule: {str(e)[:200]}"); sys.exit(1)
            except Exception as e:
                print(f"FFL_REJECTED gpu-schedule: {type(e).__name__}: {str(e)[:200]}"); sys.exit(1)
    # The pass draw and the "does it compile without the drawn passes"
    # check both work on the scheduled module: with the scheduling done
    # afterwards, that check compiled an unscheduled module for a GPU
    # target, always failed, and never recognised a pass-order artifact.
    mod_before_passes = mod
    if args.num_passes:
        pool = _pass_pool()
        chosen = rng.sample(pool, min(args.num_passes, len(pool)))
        print("FFL_PASSES " + ",".join(n for n, _ in chosen))
        tgt = _make_target(args.target)
        for name, p in chosen:
            try:
                with tgt:              # passes that need a Target context
                    mod = p(mod)
            except tvm.error.InternalError as e:
                # A pass run out of its pipeline order states its
                # precondition ("Require the target attribute", "must be
                # called after ...", "context required"): that is the pass
                # refusing the input, not an invariant broken by it. The
                # same holds for a target or module the pass cannot lower.
                if _PRECONDITION_RE.search(str(e)) or _TARGET_PRECONDITION_RE.search(str(e)):
                    print(f"FFL_REJECTED pass-precondition {name}: {str(e)[:200]}"); sys.exit(1)
                if _compiles_without_passes(mod_before_passes, args.target, args,
                                            {"tir_pipeline": args.tir_pipeline}
                                            if args.tir_pipeline != "default" else {}):
                    # the module compiles for this target with no drawn
                    # passes, so the pass was applied outside its pipeline
                    # position: its precondition, not an invariant it broke
                    print(f"FFL_REJECTED pass-order {name} (compiles with --num-passes 0): "
                          f"{str(e)[:160]}")
                    sys.exit(1)
                print(f"FFL_INTERNAL_ERROR (pass {name})"); traceback.print_exc(); sys.exit(3)
            except Exception as e:
                print(f"FFL_REJECTED pass {name}: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)

    kw = {}
    if args.tir_pipeline != "default":
        kw["tir_pipeline"] = args.tir_pipeline
    try:
        with tvm.transform.PassContext(opt_level=args.opt_level):
            lib = tvm.compile(mod, target=_make_target(target), **kw)
    except tvm.error.InternalError as e:
        if args.num_passes and _PRECONDITION_RE.search(str(e)):
            # the random passes left the module in a state the pipeline
            # does not accept: the pipeline's precondition, not a defect
            print(f"FFL_REJECTED compile-precondition: {str(e)[:200]}"); sys.exit(1)
        if args.num_passes and _compiles_without_passes(mod_before_passes, target, args, kw):
            # The module compiles when the drawn passes are left out, so
            # what failed is a pass applied outside its pipeline position,
            # not the compiler on this module. Triage used to establish
            # this by re-running every bundle with --num-passes 0; doing it
            # here keeps the class out of the findings altogether.
            print(f"FFL_REJECTED pass-order (compiles with --num-passes 0): {str(e)[:160]}")
            sys.exit(1)
        if _TARGET_PRECONDITION_RE.search(str(e)):
            # the drawn target cannot express this module (a capability it
            # does not declare, a device limit, a missing device library):
            # the target refusing the input, not a codegen defect
            print(f"FFL_REJECTED target-precondition: {str(e)[:200]}"); sys.exit(1)
        print("FFL_INTERNAL_ERROR (compile)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:
        print(f"FFL_REJECTED compile: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
    target_kind = tgt_kind      # computed once, before the pass draw
    if target_kind in GPU_KINDS:
        srcs = _device_sources(lib)
        # Only meaningful when the module has kernels to emit: a module
        # that is all host code (a Relax function calling nothing on the
        # device) legitimately produces no device module.
        problem = _check_device_source(srcs, target_kind) if _has_thread_binding(mod) else None
        if problem:
            # A GPU target that produced no device code at all: the kernel
            # was dropped somewhere in lowering. Reported as a finding, not
            # a rejection — the module compiled without an error.
            print(f"FFL_NO_DEVICE_CODE {target_kind}: {problem}"); sys.exit(5)
        if args.nvcc_arch and target_kind == "cuda" and srcs:
            nvcc = shutil.which("nvcc")
            if nvcc:
                rc, out = _cross_compile_cuda(srcs[0][1], args.nvcc_arch, nvcc)
                if rc != 0:
                    # nvcc rejecting TVM's own generated CUDA C++ is a
                    # codegen defect: the source is not the fuzzer's, it is
                    # what the compiler emitted.
                    print(f"FFL_BAD_DEVICE_SOURCE nvcc rc={rc} arch={args.nvcc_arch}\n"
                          + out[-1500:])
                    sys.exit(6)
                print(f"FFL_OK device-source nvcc {args.nvcc_arch}")
    if args.mode == "build":
        print("FFL_OK build"); return

    if target_kind in GPU_KINDS:
        # compile-only by design: no device is present
        print(f"FFL_OK build ({target_kind} compile-only)"); return
    import platform
    triple = next((p[8:] for p in target.split() if p.startswith("-mtriple=")), "")
    if triple and not triple.startswith(platform.machine()):
        print("FFL_OK build (cross target, not executed)"); return
    try:
        oob = _constant_oob(mod.script())
    except Exception:
        oob = None
    if oob:
        print(f"FFL_REJECTED out-of-bounds access in the program: {oob}"); sys.exit(1)
    try:
        rbw = _read_before_write(mod.script())
    except Exception:
        rbw = None
    if rbw:
        print(f"FFL_REJECTED read-before-write of internal buffer {rbw}"); sys.exit(1)
    try:
        entry = _entry(mod)
    except Exception as e:  # a shape of IR the detector does not know
        print(f"FFL_OK build (entry detection: {type(e).__name__}: {str(e)[:120]})"); return
    if entry is None:
        print("FFL_OK build (no runnable entry)"); return
    nrng = np.random.default_rng(args.seed)
    try:
        inputs = _random_inputs(entry[2], nrng)
    except Exception as e:
        print(f"FFL_OK build (inputs: {e})"); return
    # executed code gets TIR's bound checkers: an out-of-range access in a
    # fused kernel then fails with an error instead of a silent segfault,
    # so a signal that remains is the compiler's
    run_cfg = {"tirx.instrument_bound_checkers": True}
    try:
        with tvm.transform.PassContext(opt_level=args.opt_level, config=run_cfg):
            out = _run_once(mod, target, entry, inputs)
    except tvm.error.InternalError:
        print("FFL_INTERNAL_ERROR (run)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:
        print(f"FFL_REJECTED run: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
    if args.mode == "run":
        print("FFL_OK run"); return

    # diff: baseline at -O0 on plain llvm with the same inputs, run twice.
    # A program whose two baseline runs disagree reads memory it never
    # wrote (a reduction without an init, an alloc consumed before it is
    # produced), and no comparison against it means anything: one batch
    # filed "2.85 vs 1.2e14 (llvm opt 0 vs llvm opt 0)" as a miscompile.
    try:
        base_inputs = _random_inputs(entry[2], np.random.default_rng(args.seed))
        with tvm.transform.PassContext(opt_level=0, config=run_cfg):
            base = _run_once(mod, "llvm", entry, base_inputs)
        # Between the two runs: a fresh arena, and one call with *different*
        # inputs. A program that reads an internal buffer it never wrote (an
        # opaque block that declares a write through access_ptr and performs
        # none, a reduction without init) sees the previous call's residue,
        # so the second baseline then disagrees with the first instead of
        # matching it by luck — "1.0 vs 3.0 (core-avx2 opt 2 vs llvm opt 0)"
        # was exactly that: garbage*2+1 with garbage 0 and then 1.
        _junk = np.random.default_rng(args.seed + 1).standard_normal(
            size=random.Random(args.seed).choice([4096, 65536, 1 << 20]))
        with tvm.transform.PassContext(opt_level=0, config=run_cfg):
            _run_once(mod, "llvm", entry,
                      _random_inputs(entry[2], np.random.default_rng(args.seed + 7)))
        again_inputs = _random_inputs(entry[2], np.random.default_rng(args.seed))
        with tvm.transform.PassContext(opt_level=0, config=run_cfg):
            again = _run_once(mod, "llvm", entry, again_inputs)
    except tvm.error.InternalError:
        print("FFL_INTERNAL_ERROR (baseline run)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:
        print(f"FFL_REJECTED baseline: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
    for i, (a, b) in enumerate(zip(base, again)):
        differs = (_float_disagreement(a, b) is not None) if a.dtype.kind in "fc" \
            else (a.shape != b.shape or not np.array_equal(a, b))
        if differs:
            print(f"FFL_REJECTED nondeterministic: output {i} differs between two identical "
                  f"baseline runs (uninitialised read in the program)"); sys.exit(1)
    for i, (a, b) in enumerate(zip(out, base)):
        if a.shape != b.shape:
            print(f"FFL_MISMATCH output {i}: shape {a.shape} vs {b.shape}"); sys.exit(4)
        if a.dtype.kind in "fc":
            bad = _float_disagreement(a, b)
            if bad is not None:
                print(f"FFL_MISMATCH output {i} at {bad[0]}: {bad[1]} vs {bad[2]} "
                      f"(target {target!r} opt {args.opt_level} vs llvm opt 0)")
                sys.exit(4)
        elif not np.array_equal(a, b):
            idx = np.unravel_index(np.argmax(a != b), a.shape)
            print(f"FFL_MISMATCH output {i} at {idx}: {a[idx]} vs {b[idx]} (target {target!r} opt {args.opt_level} vs llvm opt 0)")
            sys.exit(4)
    print("FFL_OK diff")


if __name__ == "__main__":
    main()
