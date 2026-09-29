# [XLA:GPU] Reduction emitter CHECK on a pre-fused reduce that the tree-reduction rewriter never decomposed

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 reduction.cc:384] Check failed: ReductionIsRaceFree(reduction_dimensions_, analysis.device_info())
  Non-race-free reductions should have been decomposed. Did tree_reduction_rewriter run?
```

Stack: `ReductionFusion::ReductionFusion` ← `CreateReductionFusion` ← `GetFusionEmitter`
← `GpuPerformanceModelBase::EstimateFusionLaunchDimensions` ←
`GpuPerformanceModel::EstimateRunTimeForInstructionImpl`. The reduce (32768 elements per
output) is not race-free at the default limits; `TreeReductionRewriter` only rewrites
unfused reduces, so a reduce that arrives already inside a `kInput` fusion is handed to
the reduction emitter as is, and the emitter CHECKs.

## Reproducer

14 lines (from `codegen/xtile/tiling_from_block_parameters_test.cc`, with its
block-level backend config removed — the config makes no difference):

```
add {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT add = f32[] add(lhs, rhs)
}
f {
  p0 = f32[64,128,256] parameter(0)
  c0 = f32[] constant(0)
  ROOT reduce = f32[64] reduce(p0, c0), dimensions={1,2}, to_apply=add
}
ENTRY entry {
  param_0 = f32[64,128,256] parameter(0)
  ROOT fusion = f32[64] fusion(param_0),
    kind=kInput, calls=f
}
```

```bash
hlo-opt --platform=gpu --stage=buffer-assignment --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/a100_pcie_80.txtpb \
  --xla_gpu_autotune_level=0 --o=/dev/null prefused_reduce.hlo
```

`--stage=hlo` passes; the CHECK is reached from the performance model that runs after
fusion. Any spec.

## Expected

Either run the tree-reduction rewrite on reduces that already live inside fusions, or
have the emitter selection return a status for a reduction that is not race-free. A
`CHECK` with "Did tree_reduction_rewriter run?" for a module the verifier accepts is not a
user-facing error.
