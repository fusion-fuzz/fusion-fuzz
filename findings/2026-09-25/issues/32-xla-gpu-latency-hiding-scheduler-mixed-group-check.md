# [XLA:GPU] LatencyHidingScheduler CHECK-fails on a scheduling group that mixes `keep_original_sequence_order_in_group` on and off

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 latency_hiding_scheduler.cc:267] Check failed: HasKeepOriginalSequenceOrderInGroupAttribute(instr)
```

Stack: `DefaultSchedulerCore::GetNumResourcesNeededForAnnotation` ←
`SchedulingAnnotationCrossesOverlapLimit` ← `TryScheduleOneAnnotationGroup` ←
`ScheduleComputation` ← `LatencyHidingScheduler::RunImpl`. Two collective-permute
start/done pairs share `_scheduling_group_id="0"`; one pair carries
`keep_original_sequence_order_in_group="true"`, the other does not. The scheduler decides
the group keeps its original order from the first instruction it sees and then CHECKs
when another member of the group lacks the attribute.

## Reproducer

9 lines; the verifier accepts it:

```
ENTRY %entry {
  %p1 = f32[128]{0} parameter(0)
  %p2 = f32[512]{0} parameter(1)
  %cp1s = (f32[512]{0}, f32[512]{0}, u32[], u32[]) collective-permute-start(%p2), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0", keep_original_sequence_order_in_group="true"}
  %cp1d = f32[512]{0} collective-permute-done(%cp1s), frontend_attributes={_scheduling_group_id="0", keep_original_sequence_order_in_group="true"}
  %cp2s = (f32[128]{0}, f32[128]{0}, u32[], u32[]) collective-permute-start(%p1), source_target_pairs={{1,0},{0,3},{3,2}}, frontend_attributes={_scheduling_group_id="0"}
  %cp2d = f32[128]{0} collective-permute-done(%cp2s), frontend_attributes={_scheduling_group_id="0"}
  ROOT %t = (f32[512]{0}, f32[128]{0}) tuple(%cp1d, %cp2d)
}
```

```bash
hlo-opt --platform=gpu --stage=hlo-backend \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_pcie.txtpb \
  --o=/dev/null mixed_group.hlo
```

Same with the B200 spec; no other flags are needed and no GPU.

## Expected

A scheduling group whose members disagree on the attribute is a malformed annotation:
reject it in the HLO verifier or in the scheduler's annotation collection with a status
(or treat the group as not order-preserving), not with a CHECK inside scheduling.
