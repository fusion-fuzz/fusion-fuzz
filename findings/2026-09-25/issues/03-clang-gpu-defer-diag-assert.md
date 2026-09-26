# [clang][CUDA] `-fgpu-defer-diag` asserts in `VarDecl::evaluateValueImpl` on an ill-formed initialiser

**Repo:** llvm/llvm-project · **Commit:** `b9b5fb9fec7a` · **Build:** assertions enabled

## What happens

```
clang-24: clang/lib/AST/Decl.cpp:2567:
  const clang::APValue* clang::VarDecl::evaluateValueImpl(...):
  Assertion `!Init->isValueDependent()' failed.
```

## Reproducer

`defer.cu`:

```cuda
struct a { __host__ __device__ a(a &); } b;
```

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_75 \
        -fgpu-defer-diag -c -o /dev/null -std=c++20 defer.cu
```

## Expected

The clean diagnostic the compiler gives without the flag:
`error: no matching constructor for initialization of 'a'`. With `-fgpu-defer-diag` the
deferred-diagnostic pass evaluates the initialiser of a variable whose initialisation has
already failed, and asserts.

## Notes

One line of code, and the flag is the only difference between a diagnostic and an abort.
