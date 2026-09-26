# [TVM][NVPTX] Codegen emits invalid LLVM IR for a symbolic-shape kernel: `mul nsw i32 %0, i64 1024`

**Repo:** apache/tvm · **Commit:** `5b4b753` (main, 2026-09-21) · LLVM 18.1.8, `RelWithDebInfo` without `NDEBUG`

## What happens

`tvm.compile(..., target="nvptx")` fails TVM's own LLVM module verification, because the
generated IR mixes integer widths in binary operators:

```
tvm.error.InternalError: LLVM module verification failed with the following errors:
Both operands to a binary operator are not of the same type!
  %5 = mul nsw i32 %0, i64 1024
Both operands to a binary operator are not of the same type!
  %7 = add nsw i64 %6, i32 %5
...
  (src/target/llvm/codegen_llvm.cc:372, CodeGenLLVM::Verify)
```

The same module compiles cleanly for `llvm` and for `cuda`; only the NVPTX path mixes the
i32 thread index with the i64 symbolic extents.

## Reproducer

`nvptx_reshape.py`:

```python
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script import relax as R


@I.ir_module(s_tir=True)
class Module:
    @R.function
    def main(A: R.Tensor([16], "float16")):
        return R.call_tir(Module.reshape, A, out_ty=R.Tensor([2, 8], "float16"))

    @T.prim_func(s_tir=True)
    def reshape(A: T.Buffer(16, "float16"), B_handle: T.handle):
        M = T.int64()
        N = T.int64()
        B = T.match_buffer(B_handle, [M, N], dtype="float16")
        for i, j in T.grid(M, N):
            with T.sblock("compute"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi * N + vj]


# Compiles for "llvm" and for "cuda"; fails LLVM module verification for nvptx.
lib = tvm.compile(Module, target="nvptx")
print("compiled for nvptx")
```

```bash
python3 nvptx_reshape.py
```

No GPU and no scheduling primitives are involved; a `USE_LLVM=ON` build is enough.

## Expected

Valid IR: the index computation should happen in one integer width. The trigger is a
`call_tir` whose output buffer is `T.match_buffer` with `T.int64()` symbolic extents.
