# [XLA:GPU] AllReduceContiguous RET_CHECKs on all-reduces that carry a sharding

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
INTERNAL: RET_CHECK failure (xla/hlo/transforms/collectives/all_reduce_contiguous.cc:44)
  !all_reduce->has_sharding()
```

The GPU pipeline runs `AllReduceContiguous` unconditionally. A module with
`replica_count=4`, one partition and two all-reduces annotated `sharding={maximal device=N}`
keeps its sharding annotations (there is nothing for the SPMD partitioner to do with one
partition), and the pass's precondition turns into a compile failure.

## Reproducer

13 lines; the verifier accepts it:

```
HloModule m, replica_count=4
add {
  a = f32[] parameter(0)
  b = f32[] parameter(1)
  ROOT s = f32[] add(a, b)
}
ENTRY e {
  p0 = f32[128] parameter(0), sharding={maximal device=0}
  p1 = f32[128] parameter(1), sharding={maximal device=1}
  ar0 = f32[128] all-reduce(p0), replica_groups={}, to_apply=add, sharding={maximal device=0}
  ar1 = f32[128] all-reduce(p1), replica_groups={}, to_apply=add, sharding={maximal device=1}
  ROOT t = (f32[128], f32[128]) tuple(ar0, ar1), sharding={{maximal device=0}, {maximal device=1}}
}
```

```bash
hlo-opt --platform=gpu --stage=hlo \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb \
  --o=/dev/null ar_sharded.hlo
```

Exit code 1 with the status above; no flags beyond the spec are needed, and any spec
(P100, H100, B200) gives the same. Dropping `replica_count=4`, dropping the shardings,
or keeping only one all-reduce makes it compile.

## Expected

Either strip or ignore the sharding when the module is not partitioned, or skip the
all-reduce pair instead of failing the whole compilation with an INTERNAL status.
