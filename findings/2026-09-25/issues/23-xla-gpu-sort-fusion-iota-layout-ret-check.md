# [XLA:GPU] Sort emitter RET_CHECK: an iota inside a pre-existing sort fusion is left without a layout

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
INTERNAL: RET_CHECK failure (xla/backends/gpu/codegen/sort.cc:63)
  LayoutUtil::LayoutsInShapesEqual( keys_shape, sort->operand(i)->shape(),
                                    Layout::Equal().IgnoreMemorySpace().IgnoreElementSize())
```

The module arrives with the sort already inside a `kInput` fusion whose operands are a
parameter and an `iota`. After the HLO pipeline (`--stage=hlo`) the fused iota prints as
`s32[16384] iota()` — no layout — while the parameter and the sort keep `{0}`. Layout
assignment does not assign a layout to the iota inside the existing fusion, and the sort
emitter then refuses the operand-vs-keys layout comparison.

## Reproducer

18 lines (from `service/gpu/alias_info_test.cc`):

```
sorting_computation {
  %lhs_key = s32[] parameter(0)
  %rhs_key = s32[] parameter(1)
  %lhs_update_0 = s32[] parameter(2)
  %rhs_update_0 = s32[] parameter(3)
  %lhs_permutation = s32[] parameter(4)
  %rhs_permutation = s32[] parameter(5)
  ROOT %compare = pred[] compare(%lhs_key, %rhs_key), direction=LT
}
sort_fusion {
  p0 = s32[16384]{0} parameter(0)
  iota = s32[16384]{0} iota(), iota_dimension=0
  ROOT sort = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) sort(p0, iota, iota), dimensions={0}, is_stable=true, to_apply=sorting_computation
}
ENTRY main {
  p = s32[16384]{0} parameter(0)
  ROOT fusion = (s32[16384]{0}, s32[16384]{0}, s32[16384]{0}) fusion(p), kind=kInput, calls=sort_fusion
}
```

```bash
hlo-opt --platform=gpu --stage=ptx --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h200.txtpb --xla_gpu_autotune_level=0 \
  --o=/dev/null sort_iota.hlo
```

Exit code 1 with the INTERNAL status above; the verifier accepts the module. Any spec.

## Expected

Layout assignment should give every instruction inside a fusion computation a layout (or
the fusion should be re-verified with layouts after the pipeline), so that the emitter
never sees a layout-less operand.
