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
