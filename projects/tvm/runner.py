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
    """Does the module compile for the same target with no drawn passes?"""
    import tvm
    try:
        with tvm.transform.PassContext(opt_level=args.opt_level):
            tvm.compile(mod, target=_make_target(target), **kw)
        return True
    except Exception:
        return False


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

    # diff: baseline at -O0 on plain llvm with the same inputs
    try:
        base_inputs = _random_inputs(entry[2], np.random.default_rng(args.seed))
        with tvm.transform.PassContext(opt_level=0, config=run_cfg):
            base = _run_once(mod, "llvm", entry, base_inputs)
    except tvm.error.InternalError:
        print("FFL_INTERNAL_ERROR (baseline run)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:
        print(f"FFL_REJECTED baseline: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
    for i, (a, b) in enumerate(zip(out, base)):
        if a.shape != b.shape:
            print(f"FFL_MISMATCH output {i}: shape {a.shape} vs {b.shape}"); sys.exit(4)
        if a.dtype.kind in "fc":
            if not np.allclose(a, b, rtol=1e-3, atol=1e-4, equal_nan=True):
                idx = np.unravel_index(np.argmax(np.abs(a.astype("float64") - b.astype("float64"))), a.shape)
                print(f"FFL_MISMATCH output {i} at {idx}: {a[idx]} vs {b[idx]} (target {target!r} opt {args.opt_level} vs llvm opt 0)")
                sys.exit(4)
        elif not np.array_equal(a, b):
            idx = np.unravel_index(np.argmax(a != b), a.shape)
            print(f"FFL_MISMATCH output {i} at {idx}: {a[idx]} vs {b[idx]} (target {target!r} opt {args.opt_level} vs llvm opt 0)")
            sys.exit(4)
    print("FFL_OK diff")


if __name__ == "__main__":
    main()
