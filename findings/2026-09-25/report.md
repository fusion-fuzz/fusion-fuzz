# FusionFuzz — GPU-backend campaign and three-hour budget runs

**Period:** 2026-09-24 to 2026-09-26 · **Host:** 16 cores, no GPU · **Mode:** compile only

## 1. What was done

The five machine-learning compiler adapters were switched to fuzz their **GPU backends**,
compile-only. No GPU is present and nothing is ever launched: the device comes from a
target description (XLA), a module attribute (Triton), a target string (TVM) or
`--cuda-gpu-arch` (CUDA). Then each project was given a three-hour budget on the
requested command line

```bash
python3 main.py --project <pj> --setup --dataflow-fusion --state-fusion \
                --declaration-fusion --concurrency 16
```

split into three 55-minute batches, with inspection and fixes between batches.

| Project | Throughput | Fused validity | Crashing runs | Bundles | GPU share of draws |
|---|---|---|---|---|---|
| XLA | 15 tests/s | 26% | 1,194 | 465 | 70% |
| Triton | 197 tests/s | 20 → 23% | 80 | 29 | every module |
| CUDA | 5.6 tests/s | 26 → 50% | 50 | 11 | device-side compiles |
| TVM | 6.8 tests/s | 24% | 458 | 107 | 75% of build draws |
| Mojo | 2.3 tests/s | 33% | 110 | 62 | none (blocked, see §4) |

## 2. Ten bugs with verified minimal reproducers

Each has an upstream-ready issue in `issues/` and a standalone reproducer in `repro/`.
Every one was re-verified after cleaning, from a fresh process, with no fuzzer involved.

| # | Project | Signature | Reproducer |
|---|---|---|---|
| 01 | openxla/xla | `Instructions.cpp:3111 Invalid cast!` — `SliceToDynamic` with an f8 tensor as its size operand | 8 lines HLO |
| 02 | llvm/llvm-project | `SemaTemplate.cpp:7385` — `surface` partial specialisation during error recovery | 4 lines CUDA |
| 03 | llvm/llvm-project | `Decl.cpp:2567 !Init->isValueDependent()` — `-fgpu-defer-diag` | 1 line CUDA |
| 04 | llvm/llvm-project | `ToolChains/Clang.cpp:8334` — two `--cuda-gpu-arch` and a failing `ptxas` | empty file |
| 05 | modular/modular | heap corruption: `double free or corruption`, SIGSEGV | 29 lines Mojo |
| 06 | triton-lang/triton | `ArrayRef.h:247 Invalid index` under `-tritongpu-coalesce` | 41 lines MLIR |
| 07 | apache/tvm | NVPTX codegen emits `mul nsw i32 %0, i64 1024` — invalid LLVM IR | 20 lines Python |
| 08 | apache/tvm | Vulkan codegen output rejected by its own validator: `Invalid SPIR-V header` | 25 lines Python |
| 09 | apache/tvm | SPIR-V 1.4 emitted while the target declares SPIR-V 1.0 | 48 lines Python |
| 10 | apache/tvm | `jit()` + unresolved extern symbol → SIGSEGV on an unrelated call | 40 lines Python |

Ranked by what I would file first: **07** (invalid IR from a supported target, no
scheduling involved), **05** (memory corruption in a compiler), **10** (crash where an
error is expected), **01** and **04** (clean, tiny, obviously wrong), then the two SPIR-V
issues, then the three error-recovery assertions.

Not written up as issues, but recorded in the per-project reports: XLA's
`block_scaling_rewriter.cc:105/489` RET_CHECKs (fusion-only, mxfp8 scaled-dot rewriter),
the ~23 XLA crash locations that also fire on a single parent module, and Triton's
`SoftwarePipeliner.cpp:167 verify(moduleOp)` — the pipeliner producing IR that fails the
verifier.

## 3. Nine harness defects found and fixed

Watching the runs was as productive as the runs themselves. All fixed and committed.

**Efficiency.** `main.py` ignored the standalone verdicts a previous `--pre-analysis` pass
had recorded unless the flag was passed again, so a third of every pair drawn was a seed
already known to fail on its own (1,530 of CUDA's 4,439; 1,858 of XLA's 10,697; 1,426 of
TVM's 3,039). Reusing the cached verdicts took CUDA's fused validity from 26% to 50% on
the same command line.

**Deduplication.** Four adapters keyed crash signatures on the input instead of the
defect, so one bug became many "findings":

- XLA quoted the offending HLO, so one `spmd_partitioner.cc:7446` bug produced 190
  bundles in a single 55-minute batch. Now instruction names, printed instructions, array
  shapes and tuple shapes are normalised; the same work then kept 133 bundles for 376
  crashing runs.
- TVM appended the drawn target kind, so one codegen check became six bundles
  (cuda/metal/opencl/nvptx/vulkan/webgpu). Now a single `-gpu` tag.
- Mojo matched its assertion pattern with `re.S`, so it keyed on a source position printed
  earlier in the diagnostic: 30 bundles for one interpreter assertion. Fixed to a
  line-local match against a compiler source file.
- Mojo filed glibc allocator aborts as "compiler crash: no frames" — the least
  informative signature for the most serious class of bug. They are now their own kind,
  which is how finding 05 surfaced.

**False findings.** Triton's target swap could move a module to an *older* compute
capability and then report the resulting refusal ("only supported on compute capability >=
89") as a bug; the swap is now monotonic and such refusals are diagnostics. Triton's
wrapper could pair `ttng.two-ctas = true` with `ttg.num-ctas = 1`, which the Blackwell MMA
lowering asserts on — the wrapper now keeps module attributes self-consistent. TVM counted
pass-order artifacts as findings; the runner now re-compiles without the drawn passes and
records a rejection if that succeeds, which also required moving `DefaultGPUSchedule`
before the pass draw. TVM's float comparison was not elementwise, so three "mismatches"
read "nan vs nan" and one compared 1.4e-45 with -0.0.

## 4. Mojo GPU: not reachable

The compiler was rebuilt with LLVM's NVPTX and AMDGPU backends
(`llvm_configure.configure(extra_targets = ["NVPTX", "AMDGPU"])`, about an hour of Bazel)
and they are in the binary. It still cannot compile for a GPU:

1. the open-source tree registers one target backend, `HostBackend`, so
   `TargetBackendRegistry::lookup` rejects `nvptx64-nvidia-cuda` whatever LLVM carries —
   both through `--target-accelerator` and through the stdlib's
   `compile_info` / `get_gpu_target`;
2. `requireMaxForAccelerator` refuses any accelerator target unless MAX is installed.

The released wheel's compiler does have the GPU backends but ships no importable `max`
Mojo package to write kernels with. The rebuild is therefore behind
`FFL_MOJO_GPU_BACKENDS=1` and off by default.

## 5. Repository state

Eleven commits, all in `main`: two for the GPU-backend work (the four drivers plus the
TVM image and oracles, then a follow-up), one for the Mojo build flag, and eight for the
harness defects above. `output/` is git-ignored, so this directory (`findings/2026-09-25/`)
holds what is worth keeping: this report, the ten issue drafts, and the ten reproducers.

Disk: 21 GB free after reclaiming ~8 GB of my own (a dangling image from the TVM rebuild,
XLA's Bazel outputs, stale scratch). Docker reports 283 GB reclaimable in images, but
those belong to other clones and containers that are out of scope here — that needs your
call.
