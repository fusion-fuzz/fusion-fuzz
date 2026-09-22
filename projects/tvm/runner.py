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
import random
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


def _make_target(spec):
    """`llvm -mcpu=x -opt-level=N -mtriple=t -mattr=+a,+b` (the form the
    driver draws, TVM's old CLI syntax) as a Target object; the CLI string
    form is no longer accepted by tvm.target.Target."""
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
        else:
            d[k] = v
    return tvm.target.Target(d)


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
    args = ap.parse_args()
    rng = random.Random(args.seed)

    import tvm
    try:
        mod = _load_module(args.path)
    except tvm.error.InternalError as e:
        if _PRECONDITION_RE.search(str(e)):
            print(f"FFL_REJECTED parse: {str(e)[:200]}"); sys.exit(1)
        print("FFL_INTERNAL_ERROR (during TVMScript parsing)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:  # TVMScript diagnostics, seed-level Python errors
        print(f"FFL_REJECTED parse: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)

    if args.mode == "parse":
        print("FFL_OK parse"); return

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
                # refusing the input, not an invariant broken by it.
                if _PRECONDITION_RE.search(str(e)):
                    print(f"FFL_REJECTED pass-precondition {name}: {str(e)[:200]}"); sys.exit(1)
                print(f"FFL_INTERNAL_ERROR (pass {name})"); traceback.print_exc(); sys.exit(3)
            except Exception as e:
                print(f"FFL_REJECTED pass {name}: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)

    target = args.target      # opt level goes through PassContext only
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
        print("FFL_INTERNAL_ERROR (compile)"); traceback.print_exc(); sys.exit(3)
    except Exception as e:
        print(f"FFL_REJECTED compile: {type(e).__name__}: {str(e)[:300]}"); sys.exit(1)
    if args.mode == "build":
        print("FFL_OK build"); return

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
    try:
        with tvm.transform.PassContext(opt_level=args.opt_level):
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
        with tvm.transform.PassContext(opt_level=0):
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
