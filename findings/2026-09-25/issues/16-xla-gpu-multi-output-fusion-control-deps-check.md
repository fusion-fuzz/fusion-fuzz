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

## Variant: sibling fusion path (`hlo_instructions.cc:3068`)

The same defect is reached through `MultiOutputFusion::FuseSiblings` →
`HloFusionInstruction::MergeFusionInstructionIntoMultiOutput`, where the CHECK is

```
F0000 hlo_instructions.cc:3068] Check failed: instruction_to_merge->parent()->RemoveInstruction(instruction_to_merge) is OK
  (INTERNAL: RET_CHECK failure (xla/hlo/ir/hlo_computation.cc:806) ignore_safety_check || IsSafelyRemovable(instruction)
   cannot remove instruction: %b.1 = f32[32] fusion(%m), kind=kLoop, calls=%x ...)
```

Two `call`s of a computation whose two loop fusions are ordered by a control dependency
(the shape of XLA's command-buffer tests), sharing one operand, get inlined; the two
`negate` fusions of `%m` are then siblings and the pass merges one that still carries the
control edge. 23 lines:

```
x {
  a = f32[32] parameter(0)
  ROOT b = f32[32] negate(a)
}
y {
  a = f32[32] parameter(0)
  ROOT b = f32[32] add(a, a)
}
command_buffer {
  p = f32[32] parameter(0)
  q = f32[32] parameter(1)
  b = f32[32] fusion(p), kind=kLoop, calls=x
  c = f32[32] fusion(q), kind=kLoop, calls=y, control-predecessors={b}
  ROOT t = (f32[32], f32[32]) tuple(b, c)
}
ENTRY main {
  m = f32[32] parameter(0)
  n = f32[32] parameter(1)
  n2 = f32[32] parameter(2)
  call = (f32[32], f32[32]) call(m, n), to_apply=command_buffer
  call2 = (f32[32], f32[32]) call(m, n2), to_apply=command_buffer
  ROOT r = ((f32[32], f32[32]), (f32[32], f32[32])) tuple(call, call2)
}
```

```bash
hlo-opt --platform=gpu --stage=hlo \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb \
  --o=/dev/null mof_siblings.hlo
```

A single `call` compiles; the crash needs the two calls sharing `%m` (V100 and H100 specs
checked).
