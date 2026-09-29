# [Triton] `convert-nv-gpu-to-llvm` asserts on a `ttng.warp_group_dot` with bf16 × f16 operands instead of diagnosing it

**Repo:** triton-lang/triton · **Commit:** `d1fc86b` (main, 2026-09-21) · **Build:** assertions on

## What happens

```
triton-opt: lib/Conversion/NVGPUToLLVM/NVGPUToLLVMPass.cpp:467:
  Assertion `supported && "WGMMA type or shape is not supported"' failed.
```

`ttng.warp_group_dot` with an `bf16` A operand and an `f16` B operand passes the
verifier and `--convert-triton-gpu-to-llvm`; the `nvgpu.wgmma` it produces has no
matching PTX instruction and the NVGPU → LLVM lowering asserts. With both operands
`f16` (or both `bf16`) the same module lowers.

## Reproducer

12 lines (the module is upstream's `fence-insertion` test `reg_argument`, which only runs
`--triton-nvidia-gpu-fence-insertion` on it; run through the lowering it crashes):

```
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [16, 2], warpsPerCTA = [1, 8], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @reg_argument(%arg0: tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>, %arg1: tensor<128x64xf16, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %1 = ttg.local_alloc : () -> !ttg.memdesc<128x64xf16, #shared1, #smem, mutable>
    %2 = ttng.warp_group_dot %arg0, %1, %cst : tensor<128x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<128x64xf16, #shared1, #smem, mutable> -> tensor<128x64xf32, #mma>
    tt.return
  }
}
```

```bash
triton-opt wgmma_mixed.mlir --allocate-shared-memory --convert-triton-gpu-to-llvm --convert-nv-gpu-to-llvm
```

## Expected

Either the `ttng.warp_group_dot` verifier should require matching (or explicitly
supported) operand element types, or the NVGPU lowering should emit an `op.emitError`
for an unsupported WGMMA type/shape combination instead of asserting.
