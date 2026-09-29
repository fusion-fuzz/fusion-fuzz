# [XLA:GPU] GpuHloCostAnalysis CHECK-fails on a fusion whose root is a token

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 shape_util.cc:757] Check failed: shape.IsArray() || shape.IsTuple()
```

`ShapeUtil::ElementsInRecursive` is called from
`GpuHloCostAnalysis::FusionCalculateUtilizations` (`HloCostAnalysis::HandleFusion`) on the
fusion's shape, which is `token[]`: the fusion wraps an `outfeed` and returns its token.
The verifier accepts a loop fusion with a token root.

## Reproducer

8 lines:

```
fused_computation {
  token0 = token[] after-all()
  constant = u32[3]{0} constant({1,2,3})
  ROOT  outfeed = token[] outfeed(constant, token0), outfeed_shape=u32[3]{0}
}
ENTRY main {
  ROOT result = token[] fusion(), kind=kLoop, calls=fused_computation
}
```

```bash
hlo-opt --platform=gpu --stage=hlo --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_pcie.txtpb --o=/dev/null token_fusion.hlo
```

Any spec reproduces it; no GPU is needed. The CPU pipeline (`--platform=cpu --stage=hlo`)
accepts the module.

## Expected

Cost analysis should treat a token-shaped (or otherwise element-less) fusion root as zero
elements, or the GPU fusion passes should refuse to keep such a fusion, instead of a
CHECK.
