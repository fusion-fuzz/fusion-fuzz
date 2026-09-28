# [TVM][RISC-V] Vectorized store of a fixed-width `T.Broadcast` into a `float32x4` buffer ICHECKs with `+v`: "Can't broadcast between scalable and fixed length vectors"

**Repo:** apache/tvm · **Commit:** `5b4b753` (main, 2026-09-21) · LLVM 18.1.8

## What happens

```
tvm.error.InternalError: Check failed:
  (op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable) is false:
  Can't broadcast between scalable and fixed length vectors.
```

The same module compiles for x86 (`llvm`) and for AArch64 with `+sve`. Only
`riscv64-linux-gnu` with `-mattr=+v` fails: the vectorizer turns the `T.vectorized(4)`
loop into a scalable vector for RVV, then meets the explicit fixed-width
`T.Broadcast(1.0, 4)` written to a `float32x4` element and asserts instead of either
keeping the loop fixed-width or reporting that the two cannot mix.

## Reproducer

`rvv_broadcast.py`:

```python
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T

@I.ir_module
class Module:
    @T.prim_func(s_tir=True)
    def main(A: T.Buffer((4,), "float32x4"), n: T.int32):
        for i in range(n):
            for j in T.vectorized(4):
                A[j] = T.Broadcast(T.float32(1), 4)

for name, tgt in (("x86 llvm", {"kind": "llvm"}),
                  ("riscv64 +v", {"kind": "llvm", "mtriple": "riscv64-linux-gnu", "mcpu": "generic-rv64", "mattr": ["+v"]}),
                  ("aarch64 +sve", {"kind": "llvm", "mtriple": "aarch64-linux-gnu", "mattr": ["+sve"]})):
    try:
        tvm.compile(Module, target=tvm.target.Target(tgt)); print(f"{name:14s} -> OK")
    except Exception as e:
        print(f"{name:14s} -> {type(e).__name__}: {str(e).strip().splitlines()[-1][:110]}")
```

```
$ python3 rvv_broadcast.py
x86 llvm       -> OK
riscv64 +v     -> InternalError: Check failed: (op->ty.as_or_throw<PrimType>().IsScalableVector() == is_scalable) is false: Can't broadcast bet...
aarch64 +sve   -> OK
```

No random passes and no scheduling are involved; `tvm.compile` on the module as written
is enough.

## Expected

Either a compiled module (the loop is a fixed 4-wide vector of `float32x4` elements, so
a fixed-width lowering is valid) or a diagnostic. An ICHECK in the middle of the pipeline
is neither. AArch64 SVE taking the same input suggests the RVV path chooses a scalable
width where the SVE path does not.

## Notes

Found by fuzzing TVM's own test modules with cross-compilation targets; one of the two
parent test modules reproduces it on its own, so no fusion is needed.
