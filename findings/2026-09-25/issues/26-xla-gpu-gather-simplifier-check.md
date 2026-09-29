# [XLA:GPU] Loop fusion emitter CHECK on a non-simplified gather inside a kCustom fusion

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 indexing_analysis.cc:536] Check failed: GatherSimplifier::IsSimplifiedGather(gather)
  Non-simplified HLO Gather is not supported.
```

Stack: `ComputeOutputToInputIndexing` ← `emitters::PartitionedComputation` ←
`LoopFusionKernelEmitter::EmitKernelDefinition` ← `LoopFusion::CreateMLIRModule` ←
`MlirKernelEmitter::Emit`. `GatherSimplifier` runs on top-level gathers only; a gather that
arrives already inside a fusion (here with rank-1 indices and `index_vector_dim=1`) keeps
its original form, and the indexing analysis used by the loop emitter CHECKs on it.

## Reproducer

30 lines (line-reduced from `memory_space_assignment_test.cc`; every reduction step kept the
HLO verifier's acceptance as a requirement):

```
%fused_computation {
  %param_0.2 = f32[32]{0} parameter(0)
  %param_1.4 = s32[100]{0} parameter(1)
  %custom-call.1 = s32[100]{0} custom-call(s32[100]{0} %param_1.4), custom_call_target="AssumeGatherIndicesInBound", operand_layout_constraints={s32[100]{0}}
  %slice.1 = s32[32]{0} slice(s32[100]{0} %custom-call.1), slice={[0:32]}
  %reshape.7 = s32[32]{0} reshape(s32[32]{0} %slice.1)
  %transpose.5 = s32[32]{0} transpose(s32[32]{0} %reshape.7), dimensions={0}
  %gather.1 = f32[32]{0} gather(f32[32]{0} %param_0.2, s32[32]{0} %transpose.5), offset_dims={}, collapsed_slice_dims={0}, start_index_map={0}, index_vector_dim=1, slice_sizes={1}
}
%i.reduce_sub_computation {
  %rhs = s32[] parameter(1)
  %lhs = s32[] parameter(0)
}
%fused_computation.1 {
  %constant.4 = s32[] constant(0)
  %broadcast.4 = s32[100]{0} broadcast(s32[] %constant.4), dimensions={}
  %param_0.4 = s32[32]{0} parameter(0)
  %pad.1 = s32[100]{0} pad(s32[32]{0} %param_0.4, s32[] %constant.4), padding=0_68
  %constant.3 = s32[] constant(76031)
  %broadcast.3 = s32[100]{0} broadcast(s32[] %constant.3), dimensions={}
  ROOT %clamp.1 = s32[100]{0} clamp(s32[100]{0} %broadcast.4, s32[100]{0} %pad.1, s32[100]{0} %broadcast.3)
}
ENTRY %main {
  %constant = s32[] constant(0)
  %i = s32[32,1]{0,1} parameter(1)
  %o = f32[32]{0} parameter(0)
  %reduce = s32[32]{0} reduce(s32[32,1]{0,1} %i, s32[] %constant), dimensions={1}, to_apply=%i.reduce_sub_computation
  %fusion.1 = s32[100]{0} fusion(s32[32]{0} %reduce), kind=kLoop, calls=%fused_computation.1
  ROOT %fusion = f32[32]{0} fusion(f32[32]{0} %o, s32[100]{0} %fusion.1), kind=kCustom, calls=%fused_computation
}
```

```bash
hlo-opt --platform=gpu --stage=buffer-assignment --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/b200.txtpb --xla_gpu_autotune_level=0 \
  --o=/dev/null fused_gather.hlo
```

Any spec; no GPU is needed.

## Expected

Either simplify gathers inside existing fusions (run `GatherSimplifier` on fusion
computations) or fall back to the legacy emitter / return a status when the indexing
analysis cannot handle the gather.
