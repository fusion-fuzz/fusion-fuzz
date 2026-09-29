# [XLA:GPU] Multi-output fusion CHECK-fails removing a fused instruction that carries control dependencies

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 multi_output_fusion.cc:425] Check failed: computation_->RemoveInstruction(fused) is OK
  (INTERNAL: RET_CHECK failure (xla/hlo/ir/hlo_computation.cc:806)
   ignore_safety_check || IsSafelyRemovable(instruction, ...))
```

`GpuMultiOutputFusion` fuses sibling elementwise consumers of the same operands, then
removes the originals. One of them is a control predecessor of another instruction, so
`IsSafelyRemovable` refuses and the pass CHECKs instead of skipping the candidate.

## Reproducer

11 lines; the HLO verifier accepts it and the CPU backend compiles it:

```
HloModule m

ENTRY %e {
  %v1 = f32[4] parameter(0)
  %v2 = f32[4] parameter(1)
  %gt = pred[4] compare(%v1, %v2), direction=GT
  %add = f32[4] add(%v1, %v2)
  %mul = f32[4] multiply(%v1, %v2), control-predecessors={%add}
  %div = f32[4] divide(%v1, %v2), control-predecessors={%mul}
  %t = (f32[4], f32[4], f32[4]) tuple(%add, %mul, %div)
  %select = f32[4] select(%gt, %v1, %v2)
  ROOT %r = (f32[4], (f32[4], f32[4], f32[4])) tuple(%select, %t)
}
```

```bash
hlo-opt --platform=gpu --stage=hlo-backend \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb \
  --xla_gpu_autotune_level=0 --o=/dev/null mof.hlo
```

Same result with the A100 and B200 specs; no GPU is needed (the device comes from the
spec). Dropping the `control-predecessors` makes it compile; dropping the `select` (the
sibling consumer that makes `add`/`mul`/`div` multi-output-fusion candidates) also makes
it compile. Sharding annotations are irrelevant.

## Expected

Either do not fuse instructions that participate in control dependencies, or transfer the
dependencies to the fusion, as the other fusion passes do. A CHECK on a well-formed module
is neither.
