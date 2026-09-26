# [Triton] An axis-info hint attribute sized like the tensor (not like its rank) makes `-tritongpu-coalesce` index out of bounds

**Repo:** triton-lang/triton · **Commit:** `d1fc86b` (main, 2026-09-21) · prebuilt LLVM `b010a18d`, assertions enabled

## What happens

`AxisInfo::initDimVectorFromHint` expands **every element** of a `DenseElementsAttr` hint
into the per-dimension vector, without checking it against the value's rank
([`lib/Analysis/AxisInfo.cpp:1351`](https://github.com/triton-lang/triton/blob/main/lib/Analysis/AxisInfo.cpp)):

```c++
void AxisInfo::initDimVectorFromHint(Attribute attr, DimVectorT *vec) {
  if (auto int_attr = dyn_cast_or_null<IntegerAttr>(attr))
    *vec = DimVectorT(1, int_attr.getValue().getZExtValue());
  if (auto dense_attr = dyn_cast_or_null<DenseElementsAttr>(attr)) {
    auto vals = dense_attr.getValues<int>();
    *vec = DimVectorT(vals.begin(), vals.end());     // one entry per *element*
  }
}
```

So `tt.contiguity = dense<32> : tensor<128x64xi32>` on a rank-2 op yields a contiguity
vector of 8192 entries. `-tritongpu-coalesce` then builds an 8192-entry order from it and
indexes the shape with `order[0]`:

```
[tritongpu-coalesce]: Considering op: %2 = tt.load %1 : tensor<128x64x!tt.ptr<f8E4M3FN>, #ttg.blocked<...>>
[tritongpu-coalesce]: axis info of pointer: contiguity = [32, 32, 32, ... ]      (8192 entries)
[tritongpu-coalesce]: order=[8191, 8190, 8189, ... ]
[tritongpu-coalesce]: shapePerCTA=[128, 64]
triton-opt: llvm/ADT/ArrayRef.h:247: Assertion `Index < Length && "Invalid index!"' failed.
```

Stack (assertions on):

```
getNumElementsPerThread ()  lib/Dialect/TritonGPU/Transforms/Utility.cpp:188
buildCoalescedEncoding ()   lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp:64
operator() ()               lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:102
runOnOperation ()           lib/Dialect/TritonGPU/Transforms/Coalesce.cpp:86
```

`Utility.cpp:188` is `std::min(valInfo.getContiguity(order[0]), shapePerCTA[order[0]])`.
**With assertions disabled this is an out-of-bounds read**, not a diagnostic.

## Reproducer 1: your own test file, one pass

```bash
triton-opt test/TritonGPU/loop-pipeline-blackwell.mlir -split-input-file \
  -tritongpu-coalesce -o /dev/null
```

`test/TritonGPU/loop-pipeline-blackwell.mlir:441` writes the hints in the tensor-shaped
form:

```mlir
%lhs_ptrs_i = tt.addptr %lhs_ptrs, %lhs_offs {
    tt.divisibility = dense<16> : tensor<128x64xi32>,
    tt.contiguity  = dense<32> : tensor<128x64xi32>,
    tt.constancy   = dense<1>  : tensor<128x64xi32>}
  : tensor<128x64x!tt.ptr<f8E4M3FN>, #load_blocked>, tensor<128x64xi32, #load_blocked>
```

It also crashes when coalescing follows the pipeline the test itself runs
(`-tritongpu-hoist-tmem-alloc -tritongpu-assign-latencies -tritongpu-schedule-loops
-tritongpu-pipeline -tritongpu-coalesce`). Several AMD pipeline tests
(`loop-pipeline-hip.mlir`, `amd-update-async-wait-count.mlir`, ...) use the same form.

## Reproducer 2: 12 lines, self-contained

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @coalesce_hint(%base: !tt.ptr<f16>, %off: tensor<128x64xi32, #blocked>) {
    %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %ptrs, %off {tt.contiguity = dense<32> : tensor<128x64xi32>,
                                tt.divisibility = dense<16> : tensor<128x64xi32>,
                                tt.constancy = dense<1> : tensor<128x64xi32>}
        : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %v = tt.load %p : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
```

```bash
triton-opt coalesce_hint.mlir -tritongpu-coalesce -o /dev/null
```

Replacing the three hints with per-dimension values
(`tt.contiguity = dense<[1, 32]> : tensor<2xi32>`) compiles cleanly, which isolates the
cause.

## A second assertion from the same cause

If only *one* of the three hints is tensor-shaped, the sizes disagree and the `AxisInfo`
constructor asserts first (`include/triton/Analysis/AxisInfo.h:38
divisibility.size() == contiguity.size()`):

```mlir
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  tt.func public @coalesce_hint(%base: !tt.ptr<f16>, %off: tensor<128x64xi32, #blocked>) {
    %ptrs = tt.splat %base : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked>
    %p = tt.addptr %ptrs, %off {tt.contiguity = dense<32> : tensor<128x64xi32>}
        : tensor<128x64x!tt.ptr<f16>, #blocked>, tensor<128x64xi32, #blocked>
    %v = tt.load %p : tensor<128x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}
```

## Can this come from Python?

No, and that is worth stating: `tl.max_contiguous`, `tl.multiple_of` and
`tl.max_constancy` all reject a mismatched length in the frontend
(`python/triton/language/semantic.py:1847-1863`, `len(x.shape) != len(values)` →
`ValueError`), and Gluon re-exports the same builtins. The malformed hint is therefore
only reachable through IR — which is exactly what the test files above are, and what any
tool that re-runs passes over saved IR will feed in.

## Expected

Either reject the attribute or ignore it. Two candidate fixes:

1. In `initDimVectorFromHint`, take the rank from the value and ignore (or diagnose) a
   hint whose element count differs — this also fixes the `AxisInfo.h:38` variant.
2. Verify `tt.contiguity` / `tt.divisibility` / `tt.constancy` on ops
   (`lib/Dialect/Triton/IR/Dialect.cpp:91` already lists them as the known hint names),
   so malformed IR is rejected at parse time and the in-tree tests get fixed with it.
