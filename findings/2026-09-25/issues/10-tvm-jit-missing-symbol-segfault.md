# [TVM] `Executable.jit()` with an unresolved extern symbol: SIGSEGV instead of an error

**Repo:** apache/tvm · **Commit:** `5b4b753` · LLVM 18.1.8

## What happens

An IRModule holding a `tirx.call_extern` to a symbol that is not linked compiles and jits
without complaint; calling an *unrelated* function of the same module prints

```
JIT session error: Symbols not found: [ dummy_function_name ]
!!!!!!! Segfault encountered !!!!!!!
```

and the process dies on SIGSEGV (2/2 runs, exit 139).

## Reproducer

`jit_missing_symbol.py`:

```python
import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module(s_tir=True)
class Module:
    @T.prim_func(s_tir=True)
    def main():
        T.Call(tvm.ir.Op.get("tirx.call_extern"), ["dummy_function_name"], ret_ty="void")

    @T.prim_func(s_tir=True)
    def rowsum(a: T.handle, b: T.handle) -> None:
        A = T.match_buffer(a, [128])
        B = T.match_buffer(b, [])
        B_rf = T.sblock_alloc_buffer([128], elem_offset=T.int64(0))
        for i in range(128):
            with T.sblock("B_rf"):
                vi0 = T.axis.S(128, i)
                B_rf[vi0] = A[vi0]
        for i in range(128):
            with T.sblock("B"):
                vi0_1 = T.axis.R(128, i)
                with T.init():
                    B[()] = 0.0
                B[()] = B[()] + B_rf[vi0_1]


lib = tvm.compile(Module, target="llvm")
rt = lib.jit()
a = tvm.runtime.tensor(np.zeros(128, dtype="float32"))
b = tvm.runtime.tensor(np.zeros((), dtype="float32"))
rt["rowsum"](a, b)      # dies here
print("called, b =", b.numpy())
```

```bash
python3 jit_missing_symbol.py
```

## Expected

An unresolved symbol should surface as an exception from `jit()` or from the call, not a
crash. `jit()` on the same module *without* the second function returns normally, so the
failure is deferred until an unrelated call runs.
