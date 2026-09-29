# Reproducers

Each file is standalone: no fuzzer, no harness. The command for each one is in the
matching file under `../issues/`.

`clang-multi-arch-ptxas-assert.cu` is **intentionally empty** — the clang driver
assertion needs no source at all, only two `--cuda-gpu-arch` values and a `ptxas`
invocation that fails.

The `xla-*.hlo` files added on 2026-09-29 (issues 18–33; 33 is Triton) were verified that day
against the same openxla/xla `a8f4eee` build; the others were verified on 2026-09-26.

Verified against: openxla/xla `a8f4eee`, triton-lang/triton `d1fc86b`,
apache/tvm `5b4b753`, modular/modular `0103072`, llvm/llvm-project `b9b5fb9fec7a`
(clang 24.0.0git, assertions on), CUDA 13.4.59.
