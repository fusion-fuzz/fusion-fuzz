# FusionFuzz — six-hour GPU-backend runs: what is worth reporting

**Period:** 2026-09-28 14:25 → 2026-09-29 18:20 (TVM → CUDA → XLA → Triton → Mojo, six 55-minute
batches each, harness fixed between batches) · **Host:** 16 cores, no GPU · **Mode:** compile only

Command line for every batch:

```bash
python3 main.py --project <pj> --setup --dataflow-fusion --state-fusion \
                --declaration-fusion --concurrency 16 --time 3300
```

## 1. Outcome per project

| Project | Throughput | Fused validity | Bundles (6 batches) | New classes | Worth reporting |
|---|---|---|---|---|---|
| TVM | 6.8 tests/s | 24–25% | 166 | 17 | **2** (issues 11, 12) |
| CUDA (clang + nvcc) | 4–5 tests/s | 49–51% | 21 | 2 | **2** (issues 13, 14) |
| XLA | 13–14.5 tests/s | 32.7–33% | 472 | 49 | **18** (issues 15–32) + variant of 16 |
| Triton | 184–190 tests/s | 22.5% | 42 | 1 | **1** (issue 33) |
| Mojo | 2.3 tests/s | 32–34% | 45 | 2 | 0 (both are issue 05's heap corruption) |

"New classes" are crash locations not seen in the three-hour runs or in an earlier batch
(`output/tmp/worth_reporting.py`). Every new class was triaged the same way: does the
project's own verifier accept the input (verifier-rejected input is not reported), does the
crash survive with the base target and no random flags, does either parent crash alone,
and does a line-reduced (XLA/Triton/Mojo) or creduce'd (CUDA/TVM) reproducer still fire
from a fresh process. Only the survivors are below.

## 2. The 23 findings worth reporting (issues 11–33)

Each has an upstream-ready issue in `issues/` and a standalone reproducer in `repro/`
(verified 2026-09-29 against openxla/xla `a8f4eee`, triton-lang/triton `d1fc86b`,
apache/tvm `5b4b753`, llvm/llvm-project `b9b5fb9fec7a`, CUDA 13.4).

### TVM (apache/tvm)

| # | Signature | Reproducer |
|---|---|---|
| 11 | `tirx.instrument_bound_checkers` is a no-op: `InstrumentBoundCheckers` waits for `buffer_bound` attributes that no pass emits (out-of-bounds stores go undetected) | Python |
| 12 | RISC-V `+v`: vectorized store of a fixed-width `T.Broadcast` into a `float32x4` buffer ICHECKs "Can't broadcast between scalable and fixed" | Python |

### CUDA (llvm/llvm-project, clang)

| # | Signature | Reproducer |
|---|---|---|
| 13 | `-g -O0`: a `__constant__` static local in a host function asserts in `CGDebugInfo::collectVarDeclProps` (`CGDebugInfo.cpp:4663`, device pass) | 6 lines CUDA |
| 14 | `static_assert` on a typo-corrected template-id asserts "Expression evaluator can't be called on a dependent expression" (`ExprConstant.cpp:21962`) | 2 lines C++ |

### XLA (openxla/xla)

| # | Where | Signature | Reproducer |
|---|---|---|---|
| 15 | parser | `hlo_computation.cc:185` CHECK on parameter numbers that are not 0..n-1 | 3 lines |
| 16 | GPU multi-output fusion | `multi_output_fusion.cc:425` removing a fused instruction with control deps; variant via `FuseSiblings` (`hlo_instructions.cc:3068`) after inlining two calls of a control-ordered command buffer | 11 / 23 lines |
| 17 | CPU client | `cpu_client.cc:1162` "Unexpected duplicate" for an entry returning the same token twice | 5 lines |
| 18 | CPU buffer assignment | `buffer_assignment.cc:2032/2015` "Buffer doesn't have a proper live range" on an async-update inside a while loop | 28 lines |
| 19 | GPU cost analysis | `shape_util.cc:757` `ElementsInRecursive` on a token-rooted fusion (outfeed inside a loop fusion) | 8 lines |
| 20 | GPU reduction emitter | `reduction.cc:384` "Did tree_reduction_rewriter run?" — a pre-fused reduce is never decomposed | 14 lines |
| 21 | ReduceScatterCreator | `index_util.h:99` reads the pid→offset table with a device id beyond its size | 17 lines |
| 22 | CollectiveKernelStrategyAnnotator | **SIGFPE** (no CHECK): any all-reduce/all-gather whose replica group names more devices than the module has | 9 lines |
| 23 | GPU sort emitter | `codegen/sort.cc:63` RET_CHECK: an iota inside a pre-existing sort fusion never gets a layout | 18 lines |
| 24 | Triton gemm | `triton_gemm_config` with `num_ctas=3` reaches Triton's `LinearLayout.cpp:246` `isPowerOf2` assert; 4 fails cleanly | 20 lines |
| 25 | PriorityFusion | `hlo_computation.cc:917 root_found` merging fusions around a side-effecting custom-call in a while body | 50 lines |
| 26 | GPU loop emitter | `indexing_analysis.cc:536` "Non-simplified HLO Gather" inside a kCustom fusion | 30 lines |
| 27 | AllReduceContiguous | `all_reduce_contiguous.cc:44` RET_CHECK `!has_sharding()` on sharded all-reduces in a `replica_count=4` module; INTERNAL, no flags needed | 13 lines |
| 28 | shardy-xla | MLIR `Operation.h:1069` assert importing a while body rooted at its parameter next to an outfeed (`--passes` route) | 18 lines |
| 29 | shardy-xla | `module_attributes_importer.cc:351` picks a non-entry computation named `main` (`--passes` route) | 12 lines |
| 30 | block-level fusion rewriter | s4→s8 `convert` wrapped for Triton, which cannot compile it: INTERNAL with the Triton diagnostic lost (flag `--xla_gpu_experimental_enable_fusion_block_level_rewriter`) | 5 lines |
| 31 | run_hlo_module | `--isolate_instructions` on a `set-dimension-size`: CPU SIGSEGV, interpreter abort | 5 lines |
| 32 | LatencyHidingScheduler | `latency_hiding_scheduler.cc:267` on a scheduling group mixing `keep_original_sequence_order_in_group` on/off | 9 lines |

Grouped by kind: 3 are wrong-input-not-rejected (15, 22, 32 — the verifier should refuse
the module), 7 are "pre-fused HLO the pipeline assumes it produced itself" (16, 20, 23,
25, 26, 19, 24), 4 are pass preconditions turned into CHECKs (18, 21, 27, 30), 3 are
tooling (28, 29, 31) and one is the CPU client (17). 22 is the only one without a CHECK
message — a genuine divide-by-zero that a fuzzer without a signal handler classifies as
"Floating point exception" and nothing else; it appeared once in the three-hour runs and
was left unexplained then.

### Triton (triton-lang/triton)

| # | Signature | Reproducer |
|---|---|---|
| 33 | `NVGPUToLLVMPass.cpp:467` "WGMMA type or shape is not supported" assert on a `ttng.warp_group_dot` with bf16 × f16 operands (upstream's own `fence-insertion` test module, run through the lowering) | 12 lines MLIR |

### Not reported, and why

* **XLA verifier-rejected input** (the bulk of XLA's 49 new classes: `shape.h:397/807`,
  `shape_util.h:132`, the SPMD partitioner, layout assignment, algebraic simplifier,
  `hlo_evaluator_typed_visitor.h`, `literal_util.cc`, `conditional_simplifier.cc`,
  `shape_inference.cc`, …): the HLO verifier refuses the module, so the crash is only
  reachable by skipping verification.
* **XLA float→int overflow "mismatch"**: bf16 × bf16 → s32 `dot` under
  `--use_large_float_range=true` differs between the CPU backend (saturates) and the
  evaluator; HLO leaves that conversion unspecified. The analyzer now rejects such
  comparisons.
* **XLA `hlo_evaluator.cc:4465/4469 is_reflexive`**: NaN comparator (input, not bug).
* **XLA `literal.cc:775`**: `run_hlo_module` random inputs for a `set-dimension-size`
  operand can be out of range or negative; undefined input.
* **Mojo `TypeSupport.h:173` and `glibc: double free or corruption`**: both come from
  `stdlib/runtime__test_asyncrt.mojo`, which crashes the `-O0 -DASSERT=all` compiler on its
  own — issue 05's heap corruption; its line reduction lands on the already-known
  `'pop.stack_allocation' op operation destroyed but still has uses`.
* **Triton**: five known classes recur in every batch (AMD FMA dot lowering, pipeliner
  `LowerLoops.cpp:78`, `Cycle detected in call graph`, `toLinearLayout` partitioned
  layouts, `ArrayRef.h:247`); all were triaged in the three-hour runs.
* **CUDA**: saturated on the three known clang classes after batch 1; issues 13 and 14
  were the only new sites in six hours.
* **TVM**: five noise classes were removed by harness fixes during its run (pass-order
  artifacts, non-elementwise float comparison, nondeterministic outputs, out-of-bounds
  loads, read-before-write of internal buffers); what remained new were issues 11 and 12.

## 3. Harness fixes made during the runs

* `core/orchestrator.py`: a strategy that declines mid-chain (returns `None`) ends the
  chain and is recorded as a degradation naming the strategy; before, the next strategy
  read `.content` off `None` and every such pair was logged as a "Fusion error" (185 in
  the first four minutes of the Mojo run).
* `projects/xla/analyzer.py`: CPU-vs-interpreter mismatches whose integer result holds the
  type's limit values are float→int overflow and are rejected, not filed.
* `main.py`: runs without `--pre-analysis` reuse recorded seed verdicts (CUDA validity
  26% → 50%); earlier in the campaign.
* TVM runner/analyzer (during the TVM run): elementwise float comparison with a
  determinism re-run, static out-of-bounds and read-before-write guards (the TIR bound
  checker is a no-op — issue 11), precondition phrases for backend limits, per-kind
  target options.
* Triage tooling under `output/tmp/`: `worth_reporting.py` (new vs known classes, keyed
  on crash location), `xla_gpu_triage.py` / `triton_gpu_triage.py` / `cuda_gpu_triage.py`
  / `tvm_gpu_triage.py`, `line_reduce.py` for containers without creduce, and
  `red/xla_b3/{one.sh,minrun.sh}` for parent reduction and minimal-command reruns.

## 4. Repository state

Everything above is committed on `main`: the 23 issue drafts and reproducers under
`findings/2026-09-25/{issues,repro}/`, the harness fixes as separate commits. Bundles
of all thirty batches are archived under `output/bugs/<pj>.6h/batchN/` (git-ignored) with
per-batch status in `output/logs/<pj>-6h-status.txt`. Disk: 49G free.
