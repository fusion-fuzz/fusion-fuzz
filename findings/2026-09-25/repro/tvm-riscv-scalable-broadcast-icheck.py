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
