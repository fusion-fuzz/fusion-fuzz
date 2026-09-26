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
