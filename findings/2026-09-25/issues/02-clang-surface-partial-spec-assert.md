# [clang][CUDA] Assertion in `Sema::CheckTemplateArgument` while recovering from an invalid `surface` partial specialisation

**Repo:** llvm/llvm-project · **Commit:** `b9b5fb9fec7a` · **Build:** assertions enabled

## What happens

```
clang-24: clang/lib/Sema/SemaTemplate.cpp:7385:
  clang::ExprResult clang::Sema::CheckTemplateArgument(clang::NamedDecl*, ...):
  Assertion `ParamType->isPointerOrReferenceType() || ParamType->isNullPtrType()' failed.
```

The input is ill-formed and clang has already issued two diagnostics
(`use of undeclared identifier 'device_fn'`, `expected ';' after struct`) before it
asserts.

## Reproducer

`surface.cu`:

```cuda
int n = device_fn();
template <typename T, int dim = 1>
struct __attribute__((device_builtin_surface_type)) surface
struct __attribute__((device_builtin_surface_type)) surface<void, n>
```

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_60 \
        --cuda-device-only -S -o /dev/null -std=c++20 surface.cu
```

## Expected

Stop at the diagnostics already issued. Dropping
`__attribute__((device_builtin_surface_type))` from the same input gives clean
diagnostics, so the attribute's handling is what reaches the check with an unresolved
parameter type.

## Notes

Reduced from clang's own CUDA tests (a `CodeGenCUDA/surface.cu`-shaped file fused with a
module-initializer test). Crash on invalid input during error recovery.
