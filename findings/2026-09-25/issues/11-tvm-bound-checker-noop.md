# [TVM] `tirx.instrument_bound_checkers` has no effect: `InstrumentBoundCheckers` waits for `buffer_bound` attributes that no pass emits

**Repo:** apache/tvm · **Commit:** `5b4b753` (main, 2026-09-21) · LLVM 18.1.8

## What happens

With `tirx.instrument_bound_checkers = True`, an out-of-bounds read still runs and returns
whatever is next in memory. The pass is in the pipeline
(`python/tvm/s_tir/pipeline.py`, appended when the option is set) but it is a no-op:
`BoundChecker` (`src/s_tir/transform/bound_checker.cc`) only instruments accesses for
which `BoundCollector` found an `AttrStmt` with key `s_tir::attr::buffer_bound`, and
nothing in the tree creates that attribute any more — the only reference to
`buffer_bound` outside its definition in `include/tvm/s_tir/stmt.h` is the collector
that reads it. (In older TVM, `StorageFlatten(create_bound_attributes=...)` emitted it;
`FlattenBuffer` does not.)

Applying the pass directly confirms it: `InstrumentBoundCheckers()(Module)` returns the
module unchanged.

## Reproducer

`bound_checker.py`:

```python
import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T


@I.ir_module
class Module:
    @T.prim_func
    def f(A: T.Buffer((2,), "float32"), B: T.Buffer((8,), "float32")):
        for i in range(8):
            B[i] = A[i]          # reads A[2..7], past the end of A


with tvm.transform.PassContext(opt_level=0, config={"tirx.instrument_bound_checkers": True}):
    lib = tvm.compile(Module, target="llvm")
rt = lib.jit()
a = tvm.runtime.tensor(np.ones((2,), "float32"))
b = tvm.runtime.tensor(np.zeros((8,), "float32"))
rt["f"](a, b)                 # expected: a RuntimeError from the bound checker
print("ran without any error; B =", b.numpy())

# The pass alone leaves the module untouched:
changed = tvm.s_tir.transform.InstrumentBoundCheckers()(Module).script() != Module.script()
print("InstrumentBoundCheckers changed the module:", changed)
```

```
$ python3 bound_checker.py
ran without any error; B = [1.00e+00 1.00e+00 7.01e-45 1.40e-45 8.97e-44 0.00e+00 1.14e-43 0.00e+00]
InstrumentBoundCheckers changed the module: False
```

The same holds with `tir_pipeline="tirx"`, and for a constant index (`B[0, 0] = A[2, 2]`
on a `(2, 3)` buffer).

## Expected

A `RuntimeError` from the inserted assertion, as the option's name and the pass's
docstring ("Instruments bound checkers") promise. Either `FlattenBuffer` (or a small pass
before `InstrumentBoundCheckers`) should emit `buffer_bound` for every flattened buffer,
or the checker should derive the bounds from the buffer shapes itself and stop
depending on the attribute.

## Why it matters

Anyone relying on the option to catch out-of-bounds accesses in generated code gets
silent garbage instead of an error. It surfaced here because a differential test read
past a buffer and reported the leftover bytes as a miscompilation.
