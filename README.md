# Fusion Fuzz

![Tests](https://github.com/fusion-fuzz/fusion-fuzz/actions/workflows/tests.yml/badge.svg)

**Fusion Fuzz** is a language-agnostic, scalable semantic fuzzer designed to uncover deep bugs in language processors (compilers and interpreters) across a growing list of targets — PHP, CPython, Swift, Clang, MLIR, Flang, LFortran, Haskell, and more.

Unlike traditional grammar-based fuzzers (which generate code from scratch) or mutation-based fuzzers (which flip bits blindly), Fusion Fuzz operates at a higher semantic level: **program fusion** — bridging the behavior of two independent seed programs together so the result exercises interactions neither seed triggers alone.

## Program Fusion

Given two valid, pre-existing unit tests (seeds), Fusion Fuzz doesn't just concatenate their text — it bridges what they *do*. There are three fusion techniques, and most project strategies combine two or more of them:

- **Dataflow fusion** — bridges behavior at the statement/def-use level. A dataflow graph is built across both seeds' statements and topologically interleaved, so a variable defined in seed A can flow into and mutate a computation from seed B (and vice versa), while preserving def-use validity.
- **State fusion** — bridges behavior at an *intermediate runtime state*, not just the final one. Each seed is profiled for **states of interest**: program points immediately before a resource release, a type conversion, or an exception-handling boundary — places most likely to interact meaningfully with another seed's continuation, and safe to splice at without breaking syntactic/structural integrity. Fusion then grafts one seed's continuation directly into the other's state at such a point, driving the target into a combined state neither seed could reach alone. This complements dataflow fusion rather than replacing it — richer search space, deeper compiler/interpreter interactions.
- **Declaration fusion** — bridges *declarations* instead of runtime values. Many real compiler bugs never need a specific execution path to trigger — they fire the moment a declaration is compiled or instantiated. Declaration fusion finds a seed's extensible declaration expressions (a base class list, a template/generic parameter, a trait/superclass bound, an operand type constraint, ...) and makes them refer to a declaration from the *other* seed. This is especially effective on statically-verified processors, where dataflow fusion alone has comparatively little leverage.

The result is a program whose behavior — or whose declared structure — is a genuine hybrid of two independent, valid programs, which is exactly the kind of input a grammar fuzzer or a bit-flipping mutator will never construct.

### Producer-Consumer Guided Fusion

Not every pair of seeds is worth fusing — meaningful fusion needs an asymmetric relationship, where one seed contributes a semantic resource (a type it defines, a value it produces) and the other exposes a context that can consume it. Rather than pairing seeds at random, Fusion Fuzz extracts a lightweight produce/consume profile from each seed (types instantiated, symbols defined, symbols referenced) and biases parent selection toward pairs with real overlap, falling back to plain-random pairing only when no compatible match is found in budget — so fuzzing never stalls, it just loses the bias.

### Seed Migration & LLM-Assisted Fusion Adaptation

Program fusion needs a rich, well-structured seed corpus (unit tests covering language features, regression tests encoding known bug patterns) — something few languages, especially emerging ones, have accumulated independently. Fusion Fuzz instead **migrates seeds across languages**: tests already collected for one target are translated into every other target with a one-time LLM effort (`corpus/main.py`), rather than requiring each language to build a corpus from scratch.

Recognizing *states of interest* or *extensible declaration expressions* in a new language needs that language's specific syntax (resource release is `drop()` in Rust but `unset()` in PHP). Rather than hand-enumerating this for every target, Fusion Fuzz can ask an LLM to generate/refine this per-language mapping (`core/llm_mapping.py`), grounded in the project's own seed corpus.

### Validity Gap Metric

To know whether a language's mapping needs refining, Fusion Fuzz tracks the **validity gap**: the difference between how often a target's *original*, unfused seeds are accepted without error, and how often *fused* programs are. A small gap means fusion is producing programs about as often-valid as the seeds themselves; a large gap flags a language whose state/declaration mapping is producing too many syntactically-broken splices, and is a candidate for LLM-assisted refinement.

## Supported Projects

| Project | Status | Dataflow fusion | State fusion | Declaration fusion |
|---------|--------|:---:|:---:|:---:|
| ![PHP](https://img.shields.io/badge/PHP-supported-brightgreen?logo=php&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![CPython](https://img.shields.io/badge/CPython-supported-brightgreen?logo=python&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Swift](https://img.shields.io/badge/Swift-supported-brightgreen?logo=swift&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Clang](https://img.shields.io/badge/Clang-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![MLIR](https://img.shields.io/badge/MLIR-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Flang](https://img.shields.io/badge/Flang-supported-brightgreen?logo=llvm&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![LFortran](https://img.shields.io/badge/LFortran-supported-brightgreen?logo=fortran&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Haskell](https://img.shields.io/badge/GHC-supported-brightgreen?logo=haskell&logoColor=white) | **Supported** | ✅ | ✅ | ✅ |
| ![Rust](https://img.shields.io/badge/Rust-experimental-orange?logo=rust&logoColor=white) | **Experimental** | ✅ | — | ✅ |
| ![GCC](https://img.shields.io/badge/GCC-planned-lightgrey?logo=gnu&logoColor=white) | **Planned** | - | - | - |
| ![JavaScript](https://img.shields.io/badge/JavaScript-planned-lightgrey?logo=javascript&logoColor=white) | **Planned** | - | - | - |

> Go, Lean, WGSL (naga/wgslc), and Cangjie support was removed for now (was experimental/WIP) and will be reimplemented later; see git history under `projects/go`, `projects/lean`, `projects/naga`, `projects/wgslc`, `projects/cangjie` for the prior state.
>
> Rust's "declaration fusion" is item-level struct/enum/trait/impl fusion, reachable via either `--struct-fusion` or `--declaration-fusion` (the latter is an accepted alias); Rust has no state-fusion strategy yet. State/declaration fusion for the other 7 projects are each opt-in independently via `--state-fusion`/`--declaration-fusion` — pass any combination of `--dataflow-fusion`/`--state-fusion`/`--declaration-fusion`; only the ones you pass are active (defaults to dataflow fusion alone if you pass none). See [CLI Reference](#cli-reference) below.
>
> **Verification note:** Clang and CPython's state/declaration strategies were checked against real toolchains (`g++ -fsyntax-only`, `ast.parse`). PHP, Flang, LFortran, Swift, Haskell, and MLIR were verified structurally only (brace/indentation/block balance) — no local compiler for those targets was available during development. Haskell in particular has the least verification confidence given its layout-rule sensitivity; treat its `--state-fusion` output as more likely to hit `--dry-run`'s validity filter than the other targets until it's been run against real `ghc`. LFortran's strategies are `FlangFusionStrategy`/`FlangStateFusionStrategy`/`FlangDeclarationFusionStrategy` reused as-is (see `core/fusion.py`'s `LFortranFusionStrategy` and siblings) since both frontends fuzz off the same plain-Fortran seed corpus — carries the same verification confidence as flang's.

## Bugs Found

Bugs found by Fusion Fuzz are tracked at https://fusion-fuzz.github.io (updated periodically).

## Quick Start: Recommended Workflow

Fusion Fuzz has a **one-time offline pre-analysis phase** that should run *before* your first real fuzzing session for a project, and again any time you add new seeds. Skipping it doesn't break anything, but it means every fusion strategy has to recompute dataflow graphs and state-of-interest points **during** fuzzing instead of once up front, and you lose the baseline needed for the validity-gap metric. The steps below are numbered in the order you should actually run them.

### Step 0: Prerequisites

```bash
apt install -y git-lfs docker.io
pip install -r requirements.txt
```

All targets run inside Docker to protect host integrity — never build/run a target's toolchain directly on your host.

### Step 1: Build the Image and Start a Container

```bash
cd ./projects/<name>              # e.g. php, cpython, clang, mlir, flang, lfortran, swift, haskell, rust
docker build -t fusion-fuzz-<name> .
cd ../..

docker run --name fuzz-<name> -dit -m 24g \
  -v .:/home/fuzz/WorkSpace/fusion-fuzz \
  fusion-fuzz-<name>:latest

docker exec -it fuzz-<name> bash
cd /home/fuzz/WorkSpace/fusion-fuzz
```

> **Memory limit:** set `-m` to a value appropriate for your machine (e.g. `-m 16g`). This prevents OOM crashes caused by unbounded fuzzing programs.
> **Clang and MLIR** compile their toolchain from source on first `--setup` (see the project-specific notes below) — this can take hours; everything after that is normal.

Every command below runs **inside** this container, from `/home/fuzz/WorkSpace/fusion-fuzz`.

### Step 2: Parse Seeds into the Corpus

```bash
python3 main.py --project <name> --setup
```

This scans the project's seed source tree (`projects/<name>/parser.py`'s `collect_seeds`), extracts per-language metadata, and populates `projects/<name>/corpus.db`. You only need `--setup` again when the underlying seed source tree changes, or to force a full re-run of a later step (see Step 4).

### Step 3: Inject the Migrated Bug Corpus (Recommended)

```bash
python3 main.py --project <name> --setup --bug-corpus
```

`--bug-corpus` pulls pre-translated reproducers from `corpus/corpus.db` (built via **seed migration** — see [below](#seed-migration-in-detail)) into your project's corpus, so a language with few native regression tests still starts with a rich, bug-pattern-encoding seed set. Always pair it with `--setup` on a fresh corpus — `--bug-corpus` alone creates the corpus DB file but won't populate it with the project's own native seeds.

### Step 4: Run Pre-Analysis First

```bash
python3 main.py --project <name> --dry-run
```

This is the step the paper calls **seed formulation and pre-analysis**, and it's what makes fusion fast and the validity-gap metric meaningful. For every seed in the corpus, it:

1. **Executes the seed once** against the real target (5s timeout by default) and discards any seed with a non-zero return code — invalid seeds are just noise for fusion.
2. **Extracts and caches, per seed, in `corpus.db`:**
   - its **dataflow graph** (`meta['dataflows']`) — consumed by dataflow fusion so it never has to recompute def-use chains at fusion time;
   - its **states of interest** (`meta['states_of_interest']`) — program points near a resource release / type conversion / exception boundary, consumed by `--state-fusion` so it never has to re-scan a seed to find a splice point;
   - language-specific static metadata (variable types, function signatures, struct/class names, imports, ...) used by dataflow bridging and declaration fusion to build valid cross-seed references.
3. **Records the corpus's baseline valid rate** — how many of the *original*, unfused seeds this target accepts. `main.py` reports this once at startup and threads it into the orchestrator as the baseline for the **validity-gap metric** (`FuseValidRate` vs. baseline, shown live during fuzzing — see [Step 7](#step-7-monitoring-a-run)).

A seed that already has cached metadata is skipped on the next `--dry-run` (so it's safe and cheap to re-run). To force a full re-collection (e.g. after changing seed content, or after adding new state-of-interest patterns), combine it with `--setup`:

```bash
python3 main.py --project <name> --setup --dry-run
```

> **CPython only:** there's a second, optional pre-analysis pass, `--preprocessing`, that dynamically traces seeds to collect richer runtime type info (`projects/cpython/preprocessing.py`). It only runs during `--setup`:
> ```bash
> python3 main.py --project cpython --setup --preprocessing --dry-run
> ```

### Step 5: Generate or Refine the State-of-Interest Mapping

`--dry-run`'s state-of-interest extraction uses hand-seeded regex patterns per language (`core/state_analysis.py`). For a new or lightly-tuned target, or after a `--dry-run` shows a large validity gap, you can ask an LLM to generate or refine that mapping, grounded in the project's own corpus:

```bash
# Configure projects/<name>/config.yaml's `llm:` section first (provider/model/api_key)
python3 -m core.llm_mapping --project <name>                 # generate a fresh mapping
python3 -m core.llm_mapping --project <name> --refine         # refine the existing one using the validity-gap signal
```

This writes `projects/<name>/state_patterns.json`, which is merged on top of the built-in defaults automatically — no further action needed. `--refine` only does something once `output/<name>_validity_gap.json` exists (i.e., after Step 4/6) and the gap exceeds its threshold; otherwise it's a no-op.

### Step 6: Run the Fuzzer

```bash
python3 main.py --project <name> --iterations -1
```

With no fusion flags at all, this uses dataflow fusion only. Pass `--dataflow-fusion`/`--state-fusion`/`--declaration-fusion` in any combination to control exactly which techniques are active — each one you pass adds its strategy to the pool fuzzing samples from; none of them turn each other off. See the full [CLI Reference](#cli-reference) for every option and its per-project meaning:

```bash
# State fusion only (replaces the default — dataflow is NOT also enabled)
python3 main.py --project php --state-fusion

# Declaration fusion only
python3 main.py --project clang --declaration-fusion

# All three techniques in the same pool, plus a bug-corpus, plus a bounded run
python3 main.py --project cpython --bug-corpus --dataflow-fusion --declaration-fusion --state-fusion --iterations 5000
```

You do **not** need to repeat `--setup`/`--bug-corpus`/`--dry-run` on every invocation — they're idempotent and skip already-processed seeds, but there's no harm in a bare `--dry-run` before a long run just to make sure everything's cached.

### Step 7: Monitoring a Run

The live status line reports:

```
[ 0:12:34 ] Throughput: 42.1 tests/s | Bugs: 3 | FuseValidRate: 87.4% | ValidityGap: +6.2pp | PairCov: 512/4950 (10.3%, 0.7 pairs/s)
```

- **FuseValidRate** — % of fused samples this run accepted without a syntax/parse error.
- **ValidityGap** — `baseline_valid_rate - FuseValidRate` (only shown if you ran `--dry-run` first). A large, growing positive number means the active fusion strategy is producing far more invalid output than this language's seeds do on their own — see [Step 5](#step-5-generate-or-refine-the-state-of-interest-mapping).
- **PairCov** — how much of the seed corpus's pairwise combination space has been explored (producer-consumer-guided, see above).

The same numbers are persisted to `output/<name>_validity_gap.json` every 2000 iterations and at shutdown, so you can inspect them without watching the terminal.

### Step 8: Triage and Reduce

Crashes are written to `output/bugs/<name>/` as they're found (deduplicated by signature). To minimize a reproducer:

```bash
python3 main.py --reduce ./output/bugs/<name>/crash_<id>
```

See [Output Structure](#output-structure) for what each crash report contains.

---

### Seed Migration in Detail

`--bug-corpus` (Step 3) consumes a shared, pre-built `corpus/corpus.db`. To build or extend it yourself:

```bash
# Import a project's native test suite as corpus entries
python3 corpus/main.py import --project php

# Batch-translate them into another target language via LLM
python3 corpus/main.py translate-llm --target rust --source php

# Refine bad translations (e.g. ones that leaked an FFI import unsuitable for fuzzing)
python3 corpus/main.py translate-llm --target cpython --refine --filter "import ctypes" --avoid "ctypes,cffi"

# Inspect what's in there
python3 corpus/main.py stats
python3 corpus/main.py list --project php
```

Once translated, `python3 main.py --project <target> --bug-corpus` pulls the matching-language rows into that project's own corpus.

---

## Per-Project Docker Notes

The build/run pattern in [Step 1](#step-1-build-the-image-and-start-a-container) is identical for every project; the notes below only cover what's different.

#### Clang

> The first `--setup` clones llvm-project's `main` branch and compiles clang from source (installed to `projects/clang/llvm-clang-install/`), which can take a while and use significant RAM. The Dockerfile only installs the build toolchain (cmake/ninja/gcc) — there is no prebuilt clang in the image.

#### MLIR

> The first `--setup` compiles LLVM/MLIR from source, which can take several hours and requires substantial RAM. On a 32 GB machine, limit compilation parallelism (the Dockerfile does this automatically with `-j4`).

#### LFortran

> The first `--setup` clones and builds lfortran from source (installed to `/opt/lfortran`), which can take a while and use several GB of RAM per build job — see `projects/lfortran/setup.py`'s `_ninja_job_count` for why parallelism is capped by memory rather than CPU count alone. Shares its seed corpus with `projects/flang` (both are plain Fortran frontends fuzzing the same `llvm-project` test sources) instead of maintaining a separate clone.

#### PHP / CPython / Swift / Flang / Rust / Haskell

No special first-run steps beyond the standard flow above.

```bash
# Example, repeat for any <name> in the Supported Projects table:
cd ./projects/php && docker build -t fusion-fuzz-php . && cd ../..
docker run --name fuzz-php -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-php:latest
docker exec -it fuzz-php bash -c "cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project php --setup --bug-corpus --dry-run"
```

## CLI Reference

```
python3 main.py --project <name> [options]
```

**Setup & corpus**

| Flag | Default | Description |
|------|---------|-------------|
| `--project <name>` | *(required)* | Target project (folder name under `projects/`) |
| `--setup` | off | Re-parse seeds and rebuild the corpus |
| `--bug-corpus` | off | Seed corpus with pre-translated reproducers from `corpus/corpus.db` (pair with `--setup`) |
| `--preprocessing` | off | [cpython only] Run dynamic type-tracing on seeds during `--setup` |
| `--dry-run` | off | Pre-analysis: execute every seed once, cache dataflow graphs + states of interest, discard non-zero-RC seeds, record the validity-gap baseline |
| `--corpus-size <n>` | off | Sample N seeds from the loaded corpus for fusion instead of using all of them |
| `--diverse` | off | With `--corpus-size`: greedy farthest-point sampling for a dissimilar subset, instead of uniform random |
| `--save-subset <path>` | off | Save the selected corpus subset (after `--corpus-size`/`--diverse`) for reuse |
| `--load-subset <path>` | off | Load a previously saved corpus subset instead of the full project corpus |

**Fusion strategy**

| Flag | Default | Description |
|------|---------|-------------|
| `--dataflow-fusion` | on when no technique flag is given | Bridge variable linking via def-use dataflow graph. Each fusion randomly picks A→B or B→A direction. Combinable with `--state-fusion`/`--declaration-fusion` — every technique you enable is added to the same pool and one is picked at random per iteration. |
| `--state-fusion` | off | [php/cpython/clang/flang/lfortran/swift/haskell/mlir] Add a state-of-interest-driven strategy (`core/state_analysis.py`) to the pool — profiles each seed for points near a resource release/type conversion/exception boundary, then grafts one seed's continuation into the other's state there. Combinable with `--dataflow-fusion`/`--declaration-fusion`. |
| `--declaration-fusion` | off | Confuse declarations instead of dataflow: make an extensible declaration expression in one seed (base class list, template/generic parameter, trait/superclass bound, operand type constraint) refer to a declaration in the other. No runtime dataflow needed. Combinable with `--dataflow-fusion`/`--state-fusion`. [rust] alias of `--struct-fusion`. [clang] base-class + template-param injection, item nesting. [php] `implements`/`extends`/trait-use injection. [swift] protocol-conformance injection. [cpython] extra base-class injection (MRO/metaclass conflicts at class-statement time). [haskell] typeclass superclass-constraint injection. [flang/lfortran] derived-type `EXTENDS()` injection. [mlir] function-signature operand/result-type swap. |
| `--struct-fusion` | off | [rust only] Item-level fusion: nest struct/enum/trait/impl/fn definitions from one seed inside a container found in the other, plus supertrait injection, impl grafting, and generic bound injection. Same strategy as `--declaration-fusion` for this project. |

Pass none of `--dataflow-fusion`/`--state-fusion`/`--declaration-fusion` to get the default (dataflow fusion only). Pass any subset to run exactly those techniques, side by side in the same random-pick pool — e.g. `--state-fusion --declaration-fusion` (no `--dataflow-fusion`) runs only those two, never plain dataflow bridging. A combination unsupported by the target project (e.g. `--state-fusion` on `rust`, which has no state-fusion strategy) fails fast at startup with an error rather than silently doing nothing.

**Execution & output**

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations <n>` | `-1` (unlimited) | Stop after N fuzzing iterations |
| `--concurrency <n>` | from `config.yaml` | Override the number of parallel worker threads |
| `--sample-log [path]` | off | Log each sample's seeds and stdout/stderr (default path: `output/<project>_samples.log`) |
| `--setup-cov` | off | Build the project with gcov (no sanitizers) for coverage measurement |
| `--gcov` | off | After fuzzing, collect and print gcov line coverage |

**Standalone / crash triage**

| Flag | Default | Description |
|------|---------|-------------|
| `--reduce <bug_dir>` | — | Standalone: minimize a crash reproducer via delta debugging |
| `--signature <sig>` | — | Override the crash signature string used by `--reduce` |

## Output Structure

```text
output/
├── bugs/
│   └── <project>/
│       └── crash_<id>.md              # Metadata, logs, and reproduction content
├── <project>_validity_gap.json        # Baseline vs. fused valid rate, updated live (see Step 7)
├── <project>_samples.log              # Only with --sample-log
└── ...
projects/<project>/
├── corpus.db                          # SQLite corpus DB (seeds + cached pre-analysis metadata)
└── state_patterns.json                # Only if you ran core/llm_mapping.py (Step 5)
```

Each crash report (`crash_<id>.md`) contains:
- **Metadata:** Exit code, execution duration, crash signature
- **Logs:** Full `STDOUT` and `STDERR`
- **Reproduction:** The exact fused input that triggered the crash

To minimize a crash reproducer after the fact:
```bash
python3 main.py --reduce ./output/bugs/php/crash_<id>
```

## Architecture

Fusion Fuzz is structured around the following decoupled components:

### 1. Orchestrator (`core/orchestrator.py`)
The central fuzzing loop. Manages a dynamic thread pool, monitors for stalled workers, deduplicates crashes by signature (e.g., from AddressSanitizer output), tracks the validity-gap metric (baseline vs. fused valid rate) and persists it to `output/<project>_validity_gap.json`, and runs each iteration in an isolated temporary directory.

### 2. Parent Selection (`core/coverage.py`, `core/resource_matching.py`)
`PairwiseCoverageMatrix` tracks which seed pairs have already been fused and prefers unexplored combinations. `core/resource_matching.py` extracts a lightweight produce/consume resource profile per seed (from cached pre-analysis metadata, or raw token overlap as a fallback) and biases selection toward producer/consumer-compatible pairs — the **producer-consumer guided fusion** design — falling back to plain-unfused, then fully random, sampling so fuzzing never stalls.

### 3. Drivers (`core/driver.py`, `projects/*/driver.py`)
Adapters that abstract target execution. Three layers:
- **`BaseDriver`** — CLI execution, signal analysis, and crash detection. All drivers inherit from this.
- **`DockerDriver`** *(extends `BaseDriver`)* — For targets running in persistent Docker containers. Handles container lifecycle, seed file transfer via a shared `.ffl_tmp/` volume mount, and the write→exec→cleanup→result loop. Subclasses only implement `_build_exec_cmd()`.
- **Project drivers** (e.g., `PHPDriver`, `HaskellDriver`) — Override methods for target-specific logic (e.g., parsing `.phpt` headers, import hoisting, custom crash signatures).

### 4. Seed Parsers & Pre-Analysis (`core/parser.py`, `core/dryrun.py`, `projects/*/parser.py`)
- **`BaseParser`**/project parsers — scan source trees, extract metadata, and populate a per-project `corpus.db` (SQLite).
- **`core/dryrun.py`** — the **seed formulation and pre-analysis** pass run by `--dry-run` (see [Step 4](#step-4-run-pre-analysis-first)): executes every seed once, caches its dataflow graph and, via `core/state_analysis.py`, its states of interest, and records the corpus's baseline valid rate.

### 5. Fusion Engine (`core/fusion.py`, `core/state_analysis.py`)
Implements **program fusion** — bridging two seeds' behavior or declarations into one novel input:
- **Dataflow fusion** (`GenericDataflowStrategy`) — builds a def-use dependency DAG across both seeds' statements and topologically merges them.
- **State fusion** (`--state-fusion`, `*StateFusionStrategy` classes) — profiles each seed for states of interest and grafts a donor's continuation into a host's state-of-interest point using `core/state_analysis.py`'s language-agnostic pattern matching, safety checks, and graft primitive. Combinable with `--dataflow-fusion`/`--declaration-fusion`, not exclusive with them — every technique flag you pass adds its strategy to the same randomly-sampled pool. A separate, older statement/effectful mechanism (PHP's dependency-graph statement interleave, Haskell's forkIO race) still exists internally (`fuse_all()` on the affected classes) but isn't wired to any CLI flag anymore — `--state-fusion` refers only to the profiled mechanism now.
- **Declaration fusion** (`*DeclarationFusionStrategy` classes, `--declaration-fusion`) — injects a donor-declared symbol into a host's extensible declaration expression (base class list, generic bound, operand type, ...), reusable across languages via shared brace/indentation-aware splitting helpers.
- **Language-specific strategies** layer further semantic-aware splicing on top — e.g. PHP class/property instrumentation, MLIR phase-directed bug primitives.

### 6. Mutation Engine (`core/mutation.py`)
- **`BaseMutator`** — Generic mutations: arithmetic/logical operators, integer constants.
- **Project mutators** (e.g. `PHPMutator`, `CPythonMutator`, `RustMutator`, `CangjeMutator`) — Language-specific mutations: e.g. `PHP_INT_MAX`, magic constants, variable replacement.

### 7. LLM Components (`core/llmgen.py`, `core/llm_mapping.py`, `corpus/main.py`)
- **`core/llmgen.py`** — shared LLM client (gemini/openai/vllm/ollama/deepseek): provider-agnostic API calls plus `translate()`/`refine()`, used by seed migration below. (An earlier from-scratch seed-generation mode was removed as dead code — it was never wired into the fuzzing loop.)
- **`corpus/main.py`** — **seed migration**: translates a project's native test suite into every other target language, building the shared `corpus/corpus.db` consumed by `--bug-corpus`.
- **`core/llm_mapping.py`** — **LLM-assisted fusion adaptation**: generates/refines each project's `state_patterns.json` (the per-language state-of-interest regex mapping), gated by the validity-gap signal when run with `--refine`.

## Adding a New Project

Open an issue and we will look into adding support.
