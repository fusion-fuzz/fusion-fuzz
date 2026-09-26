#H_e39bddef_load_blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#H_e39bddef_scales = #ttg.linear<{register = [[0, 1], [0, 2], [32, 0], [64, 0], [0, 4]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[0, 0], [0, 0]], block = []}>
#H_e39bddef_shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#H_e39bddef_barrier_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#H_e39bddef_smem = #ttg.shared_memory
#H_e39bddef_tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 64, colStride = 1>
module attributes {"ttg.num-warps" = 4 : i32, ttg.target = "cuda:100", "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
tt.func public @H_e39bddef_load_into_async_mma(
  %lhs_ptrs: tensor<128x64x!tt.ptr<f8E4M3FN>, #H_e39bddef_load_blocked>,
  %scale_ptrs: tensor<128x8x!tt.ptr<i8>, #H_e39bddef_load_blocked>,
  %tmem: !ttg.memdesc<128x64xf32, #H_e39bddef_tmem, #ttng.tensor_memory, mutable>,
  %barrier: !ttg.memdesc<1xi64, #H_e39bddef_barrier_shared, #H_e39bddef_smem, mutable>,
  %rhs_shared: !ttg.memdesc<64x64xf8E4M3FN, #H_e39bddef_shared, #H_e39bddef_smem>,
  %n_tiles: i32
) {
  %true = arith.constant true
  %c0_i32 = arith.constant 0 : i32
  %c64_i32 = arith.constant 64 : i32
  %cst = arith.constant dense<0> : tensor<64x8xi8, #H_e39bddef_scales>
  %rhs_scales = ttng.tmem_alloc %cst : (tensor<64x8xi8, #H_e39bddef_scales>) -> !ttg.memdesc<64x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
  scf.for %i = %c0_i32 to %n_tiles step %c64_i32 : i32 {
    %lhs_offs = tt.splat %i : i32 -> tensor<128x64xi32, #H_e39bddef_load_blocked>
    %lhs_ptrs_i = tt.addptr %lhs_ptrs, %lhs_offs {tt.divisibility = dense<16> : tensor<128x64xi32>, tt.contiguity = dense<32> : tensor<128x64xi32>, tt.constancy = dense<1> : tensor<128x64xi32>} : tensor<128x64x!tt.ptr<f8E4M3FN>, #H_e39bddef_load_blocked>, tensor<128x64xi32, #H_e39bddef_load_blocked>
    %lhs = tt.load %lhs_ptrs_i : tensor<128x64x!tt.ptr<f8E4M3FN>, #H_e39bddef_load_blocked>
    %lhs_shared = ttg.local_alloc %lhs : (tensor<128x64xf8E4M3FN, #H_e39bddef_load_blocked>) -> !ttg.memdesc<128x64xf8E4M3FN, #H_e39bddef_shared, #H_e39bddef_smem>
    %scales_offs = tt.splat %i : i32 -> tensor<128x8xi32, #H_e39bddef_load_blocked>
    %scales_ptrs_i = tt.addptr %scale_ptrs, %scales_offs {tt.divisibility = dense<16> : tensor<128x8xi32>, tt.contiguity = dense<32> : tensor<128x8xi32>, tt.constancy = dense<1> : tensor<128x8xi32>} : tensor<128x8x!tt.ptr<i8>, #H_e39bddef_load_blocked>, tensor<128x8xi32, #H_e39bddef_load_blocked>
    %scales = tt.load %scales_ptrs_i : tensor<128x8x!tt.ptr<i8>, #H_e39bddef_load_blocked>
    %scales_cvt = ttg.convert_layout %scales : tensor<128x8xi8, #H_e39bddef_load_blocked> -> tensor<128x8xi8, #H_e39bddef_scales>
    %scales_tmem = ttng.tmem_alloc %scales_cvt : (tensor<128x8xi8, #H_e39bddef_scales>) -> !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>
    ttng.tc_gen5_mma_scaled %lhs_shared, %rhs_shared, %tmem, %scales_tmem, %rhs_scales, %true, %true lhs = e4m3 rhs = e4m3, %barrier[%true] {is_async} :
      !ttg.memdesc<128x64xf8E4M3FN, #H_e39bddef_shared, #H_e39bddef_smem>,
      !ttg.memdesc<64x64xf8E4M3FN, #H_e39bddef_shared, #H_e39bddef_smem>,
      !ttg.memdesc<128x64xf32, #H_e39bddef_tmem, #ttng.tensor_memory, mutable>,
      !ttg.memdesc<128x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>,
      !ttg.memdesc<64x8xi8, #ttng.tensor_memory_scales_encoding<>, #ttng.tensor_memory>,
      !ttg.memdesc<1xi64, #H_e39bddef_barrier_shared, #H_e39bddef_smem, mutable>
  }
  tt.return
}
}
