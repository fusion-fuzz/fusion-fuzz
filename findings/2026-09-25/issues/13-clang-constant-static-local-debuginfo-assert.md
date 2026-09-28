# [clang][CUDA] `-g -O0`: a `__constant__` static local in a host function asserts in `CGDebugInfo::collectVarDeclProps` ("Region stack mismatch, stack empty")

**Repo:** llvm/llvm-project · **Commit:** `b9b5fb9fec7a` · **Build:** assertions enabled · CUDA 13.4 headers

## What happens

```
clang-24: clang/lib/CodeGen/CGDebugInfo.cpp:4663:
  void clang::CodeGen::CGDebugInfo::collectVarDeclProps(...):
  Assertion `!LexicalBlockStack.empty() && "Region stack mismatch, stack empty!"' failed.
```

Sema accepts a `__constant__` variable declared `static` inside a plain host function. On
the device side that function is never emitted, but its `__constant__` local is a device
global and is, and with `-g` its debug info is collected against the enclosing function's
lexical block — which was never opened. The abort is in the device compilation pass, so a
normal `clang++ -x cuda -c -O0 -g` compile (host and device) dies too.

## Reproducer

`constant_local.cu` (6 lines, no headers):

```cuda
struct S {};
S f() {
  __constant__ static S s;
  return s;
}
void g() { f(); }
```

```bash
clang++ -x cuda --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_89 \
        --cuda-device-only -c -O0 -g -o /dev/null constant_local.cu
```

| Variation | Result |
|---|---|
| `--cuda-device-only -c -O0 -g` | assertion |
| `-c -O0 -g` (host + device) | assertion |
| `--cuda-device-only -c -O2 -g` | compiles |
| `--cuda-device-only -c -O0` (no `-g`) | compiles |
| `--cuda-host-only -c -O0 -g` | compiles |
| `__device__ void g() { f(); }` as the caller | compiles |

A function template and a `__host__ __device__` function behave the same way. The
original hit was libcu++'s `variant_test_helpers.h` / `type_id.h`, whose
`makeTypeIDImp<T>()` keeps a function-local `__constant__ static const TypeID` under
`__CUDA_ARCH__` and is called from host test code — so real code compiles this pattern
with `-g` today.

## Expected

Either a Sema diagnostic (a `__constant__` variable cannot live in a host function's
scope) or debug info that tolerates a device global whose enclosing function is not
emitted. An assertion in codegen is neither.
