# [XLA:GPU] SIGFPE in CollectiveKernelStrategyAnnotator when a replica group names more devices than the module has

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

The compiler dies with SIGFPE (integer division by zero); there is no CHECK message.
`gdb` backtrace:

```
#0  __gnu_cxx::__ops::_Iter_negate<xla::gpu::IsAllReplicasLocal(long, Span<ReplicaGroup const>, ...)::lambda>
#1  xla::gpu::IsAllReplicasLocal(long, Span<ReplicaGroup const>, CollectiveOpGroupMode, DeviceAssignment const*, ...)
#2  xla::gpu::IsAllReplicasLocal(GpuTopology const&, HloInstruction const&, DeviceAssignment const*)
#3  xla::gpu::BuildAllReduceInfo(bool, bool, GpuTopology const&, HloAllReduceInstruction const*, DeviceAssignment const*)
#4  xla::gpu::CollectiveKernelStrategyAnnotator::RunImpl
#10 xla::gpu::(anonymous namespace)::RunPostFusionPasses
#11 xla::gpu::GpuCompiler::OptimizeHloModule
```

The module config has one replica and one partition; the all-reduce's replica group is
`{0,1}`. The verifier accepts that (it does not compare group ids with the module's
device count), and `IsAllReplicasLocal` then divides by a device count derived from the
one-device topology.

## Reproducer

9 lines:

```
sum {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY main {
  p = f32[8] parameter(0)
  ROOT ar = f32[8] all-reduce(p), replica_groups={{0,1}}, to_apply=sum
}
```

```bash
hlo-opt --platform=gpu --stage=hlo --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/gb200.txtpb --o=/dev/null ar2.hlo
```

`replica_groups={}` and `replica_groups={{0}}` compile; any group with a second device
crashes. Same on H100. `channel_id`/`use_global_device_ids` make no difference, and an
`all-gather` with `replica_groups={{0,1}}` dies the same way (`BuildAllReduceInfo` is
reached for it too).

## Expected

A module whose replica groups do not fit the module's replica/partition count should be
rejected by the verifier or by the pass with a status; the pass must not divide by zero.
