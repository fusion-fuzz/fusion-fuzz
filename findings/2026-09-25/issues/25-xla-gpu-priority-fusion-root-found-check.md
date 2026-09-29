# [XLA:GPU] PriorityFusion CHECK-fails (`root_found`) merging fusions around a side-effecting custom-call in a while body

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 hlo_computation.cc:917] Check failed: root_found
```

Stack: `HloComputation::set_root_instruction` ←
`HloInstruction::ReplaceAllUsesWithDifferentShape` ← `ReplaceAllUsesWith` ←
`HloFusionInstruction::MergeFusionInstruction` ← `PriorityFusion::Fuse` ←
`PriorityFusion::RunImpl`. While merging a producer fusion into its consumer, the pass
replaces uses of the producer and ends up setting a root that is not in the computation.
The body contains a zero-operand `custom-call` (a side effect) feeding a loop fusion and a
`call`; the same custom-call is what the tree of fusions is being merged around.

## Reproducer

50 lines (line-reduced from `backends/gpu/transforms/dynamic_slice_analysis_test.cc`;
every reduction step kept the HLO verifier's acceptance as a requirement):

```
plus_one {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  one = s32[] constant(1)
  sum = s32[] add(p0, one)
  ROOT result = (s32[], s32[]) tuple(sum, p1)
}
identity {
  ROOT p0 = s32[] parameter(0)
}
remainder {
  p0 = s32[] parameter(0)
  four = s32[] constant(4)
  ROOT result = s32[] remainder(p0, four)
}
call_body {
  p0 = s32[] parameter(0)
  p1 = s32[] parameter(1)
  p2 = s32[] parameter(2)
  sum = s32[] add(p0, p2)
  nested = (s32[], s32[]) fusion(p1, sum),
      kind=kLoop, calls=plus_one
  ROOT result = s32[] get-tuple-element(nested), index=0
}
body {
  p0 = (s32[], s32[]) parameter(0)
  ivar = s32[] get-tuple-element(p0), index=0
  side_effect = s32[] custom-call(), custom_call_target=""
  derived = s32[] fusion(ivar), kind=kLoop, calls=remainder
  nested_call = s32[] call(side_effect, derived, ivar),
      to_apply=call_body
  invalid = s32[] fusion(side_effect), kind=kLoop, calls=identity
  one = s32[] constant(1)
  next_ivar = s32[] add(ivar, one)
  use = s32[] add(nested_call, invalid)
  ROOT result = (s32[], s32[]) tuple(next_ivar, use)
}
condition {
  p0 = (s32[], s32[]) parameter(0)
  ivar = s32[] get-tuple-element(p0), index=0
  five = s32[] constant(5)
  ROOT result = pred[] compare(ivar, five), direction=LT
}
ENTRY main {
  zero = s32[] constant(0)
  init = (s32[], s32[]) tuple(zero, zero)
  ROOT while = (s32[], s32[]) while(init),
      condition=condition, body=body,
      backend_config={"known_induction_variable":{"tuple_index":"0"}}
}
```

```bash
hlo-opt --platform=gpu --stage=hlo-backend --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb --xla_gpu_autotune_level=0 \
  --o=/dev/null priority_fusion_root.hlo
```

Also with `rtx6000pro.txtpb`; no GPU is needed.

## Expected

`MergeFusionInstruction` should not attempt to replace the root of a computation it is
not merging into, or `PriorityFusion` should skip this producer/consumer pair; a CHECK on
a verifier-accepted module is neither.
