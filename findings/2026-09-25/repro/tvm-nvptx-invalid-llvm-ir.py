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
