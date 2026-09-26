import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(
        x: R.Tensor((2, 3, 28, 28), dtype="float32"),
        w: R.Tensor((4, 3, 3, 3), dtype="float32"),
        begin: R.Tensor((4,), dtype="int64"),
        end: R.Tensor((4,), dtype="int64"),
        strides: R.Tensor((4,), dtype="int64"),
    ) -> R.Tensor(None, dtype="float32", ndim=4):
        with R.dataflow():
            lv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
            lv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
            gv: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                lv,
                lv1,
                strides=[1, 1],
                padding=[0, 0, 0, 0],
                dilation=[1, 1],
                groups=1,
                data_layout="NHWC",
                kernel_layout="OHWI",
                out_layout="NHWC",
                out_dtype="float32",
            )
            lv2: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                gv, axes=[0, 3, 1, 2]
            )
            gv2 = R.dynamic_strided_slice(lv2, begin, end, strides)
            R.output(gv2)
        return gv2

    @R.function
    def main_d(
        x: R.Tensor((2, 3, 4), dtype="float32"),
        indices: R.Tensor((2,), dtype="int64"),
    ) -> R.Tensor((2, 2, 4), dtype="float32"):
        R.func_attr({"num_input": 2})
        with R.dataflow():
            lv: R.Tensor((2,), dtype="int32") = R.astype(indices, dtype="int32")
            gv: R.Tensor((2, 2, 4), dtype="float32") = R.take(x, lv, axis=1, mode="fast")
            R.output(gv)
        return gv


# the binary is SPIR-V 1.4 while max_spirv_version declares 1.0
target = tvm.target.Target({'kind': 'vulkan', 'max_threads_per_block': 1024, 'thread_warp_size': 16, 'supports_float16': True, 'supports_float64': True, 'supports_int8': True, 'supports_int16': True, 'supports_int64': True, 'supports_8bit_buffer': False, 'supports_16bit_buffer': True, 'supports_storage_buffer_storage_class': True, 'supports_integer_dot_product': True, 'supports_cooperative_matrix': True, 'max_spirv_version': 66560})
lib = tvm.compile(Module, target=target)
print("compiled for vulkan")
