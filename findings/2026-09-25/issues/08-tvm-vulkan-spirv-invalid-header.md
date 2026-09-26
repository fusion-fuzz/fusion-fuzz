# [TVM][Vulkan] Generated SPIR-V is rejected by TVM's own validator: `Invalid SPIR-V header`

**Repo:** apache/tvm · **Commit:** `5b4b753` · `USE_VULKAN=ON`, `USE_KHRONOS_SPIRV=/usr` (SPIRV-Tools 2022.1), LLVM 18

## What happens

```
tvm.error.InternalError: Check failed: res == SPV_SUCCESS (-4 vs. 0) :
  index=0 error:Invalid SPIR-V header.
  (src/backend/vulkan/codegen/spirv_utils.cc:100, SPIRVTools::ValidateShader)
```

TVM generates the SPIR-V and hands it to `spvValidate`, which rejects the header, so the
module never reaches a device. No GPU is needed to reproduce.

## Reproducer

`spirv_header.py`:

```python
import tvm
from tvm.script import ir as I
from tvm.script import tirx as T
from tvm.script import relax as R

@I.ir_module
class Module:
    @R.function
    def main(data: R.Tensor((4, 2), dtype="float32")) -> R.Tensor((3, 2), dtype="float32"):
        R.func_attr({"num_input": 1})
        with R.dataflow():
            lv: R.Tensor((3, 2), dtype="float32") = R.zeros(R.shape([3, 2]), dtype="float32")
            lv1: R.Tensor((4, 1), dtype="int32") = R.expand_dims(
                R.const([0, 0, 1, 2], "int32"), axis=[1]
            )
            gv: R.Tensor((3, 2), dtype="float32") = R.scatter_nd(lv, lv1, data, reduction="add")
            R.output(gv)
        return gv

    @T.prim_func(s_tir=True)
    def single_elementwise_d(A: T.Buffer((128, 128), "float32"), B: T.Buffer((128, 128), "float32")):
        for i, j in T.grid(128, 128):
            with T.sblock("B"):
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = A[vi, vj] * 2.0


# spvValidate rejects the generated binary: 'Invalid SPIR-V header.'
target = tvm.target.Target({'kind': 'vulkan', 'max_threads_per_block': 256, 'thread_warp_size': 16, 'supports_float16': True, 'supports_float64': False, 'supports_int8': True, 'supports_int16': True, 'supports_int64': True, 'supports_8bit_buffer': True, 'supports_16bit_buffer': True, 'supports_storage_buffer_storage_class': True, 'supports_integer_dot_product': False, 'supports_cooperative_matrix': True, 'max_spirv_version': 67584})
lib = tvm.compile(Module, target=target)
print("compiled for vulkan")
```

```bash
python3 spirv_header.py
```

The module is a Relax `scatter_nd` beside an s_tir elementwise PrimFunc.

## Expected

Either valid SPIR-V or a diagnostic naming what the Vulkan backend cannot express. A
codegen whose output its own validator calls malformed is a codegen bug.
