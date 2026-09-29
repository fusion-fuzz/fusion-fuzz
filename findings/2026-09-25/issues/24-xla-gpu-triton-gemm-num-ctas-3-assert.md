# [XLA:GPU] A Triton gemm config with num_ctas=3 asserts inside Triton instead of being rejected

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
hlo-opt: external/+third_party_ext+triton/lib/Tools/LinearLayout.cpp:246:
  static LinearLayout mlir::triton::LinearLayout::strided1D(int32_t, int32_t, StringAttr, StringAttr):
  Assertion `llvm::isPowerOf2_32(size)' failed.
```

`triton_gemm_config.num_ctas` is passed straight into Triton, whose layouts require a
power-of-two CTA count. With `num_ctas=4` the same module fails cleanly
(`INTERNAL: Failed to compile Triton kernel … 'ttng.tc_gen5_mma' op RHS CTASplit along K
should be 1`); with `3` the process aborts.

## Reproducer

20 lines (the config is the one `convert_triton_gemm_config_test.cc` uses as input):

```
dot {
  lhs = f32[8192,512] parameter(0)
  rhs = f32[512,512] parameter(1)
  dot = f32[8192,512] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT transpose = f32[512,8192] transpose(dot), dimensions={1,0}
}
ENTRY entry {
  p0 = f32[8192,512] parameter(0)
  p1 = f32[512,512] parameter(1)
  ROOT fusion = f32[512,8192] fusion(p0, p1),
    kind=kCustom, calls=dot, backend_config={
      "fusion_backend_config": {
        "kind":"__triton_gemm",  "triton_gemm_config": {
          "block_m":"64", "block_n":"256", "block_k":"32",
          "num_stages":"5", "num_warps":"4", "num_ctas":"3"
        }
      }
    }
}
```

```bash
hlo-opt --platform=gpu --stage=ptx --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/gb200.txtpb --xla_gpu_autotune_level=0 \
  --o=/dev/null gemm_ctas3.hlo
```

Same on H100 (`h100_sxm.txtpb`); no GPU is needed.

## Expected

Validate `triton_gemm_config` (num_ctas a power of two, and whatever else Triton requires)
when the config is read, and return an INVALID_ARGUMENT / INTERNAL status instead of
letting Triton assert.
