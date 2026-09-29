# [XLA:GPU] ReduceScatterCreator CHECK-fails indexing an offset table with a device id beyond its size

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 index_util.h:99] Check failed: multi_index[i] < shape.dimensions(i) (7 vs. 4)
  indexing beyond extent in dimension 0
```

Stack: `IndexUtil::MultidimensionalIndexToLinearIndex` ← `LiteralBase::Piece::Get` ←
`GetIntegralAsS64` ← `MatchWithDynamicSlice` ← `MatchReduceScatter` ←
`ReduceScatterCreator`. The pattern matcher for `all-reduce` + `dynamic-slice(table[pid])`
reads the constant table at every device id named in the replica groups; the groups name
devices 0–7, the table has 4 entries, and the read is unchecked.

## Reproducer

17 lines; the HLO verifier accepts it (the module config has the default single
partition, which the verifier does not cross-check against the replica groups):

```
%sum {
  %a = f32[] parameter(0)
}
ENTRY %AllReduce {
  %param = f32[32,8,128]{2,1,0} parameter(0)
  %all-reduce = f32[32,8,128]{2,1,0} all-reduce(%param),
    replica_groups={{1,3,2,0},{7,5,6,4}}, to_apply=%sum, channel_id=1, use_global_device_ids=true
  %pid = u32[] partition-id()
  %pid_table = s32[4]{0} constant({3,0,2,1})
  %offset = s32[1] dynamic-slice(%pid_table, %pid), dynamic_slice_sizes={1}
  %reshape = s32[] reshape(%offset)
  %shard_size = s32[] constant(8)
  %mul = s32[] multiply(%reshape, %shard_size)
  %zero = s32[] constant(0)
  ROOT %dynamic-slice = f32[8,8,128] dynamic-slice(%all-reduce, %mul, %zero, %zero),
    dynamic_slice_sizes={8,8,128}
}
```

```bash
hlo-opt --platform=gpu --stage=hlo --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/gb200.txtpb --o=/dev/null rs_table.hlo
```

Any spec. With an 8-entry table the CHECK goes away and the module instead dies with the
SIGFPE of the next issue.

## Expected

`MatchWithDynamicSlice` should bail out (no match) when a device id is outside the table,
as it does for other shape mismatches, rather than CHECK inside the literal accessor.
