# Fusion Fuzz

![Tests](https://github.com/fusion-fuzz/fusion-fuzz/actions/workflows/tests.yml/badge.svg)

**Fusion Fuzz** is a scalable, semantic fuzzer designed to uncover deep bugs in language processors (compilers and interpreters) such as Rust, Swift, CPython, and more.

Unlike traditional grammar-based fuzzers (which generate code from scratch) or mutation-based fuzzers (which flip bits blindly), Fusion Fuzz operates at a higher semantic level. It generates high-quality inputs by **fusing** two or more valid, pre-existing unit tests (seeds) via dataflow interleaving — producing novel programs that exercise complex interactions the original tests never reached individually.

## Supported Projects

| Project | Status | Notes |
|---------|--------|-------|
| ![PHP](https://img.shields.io/badge/PHP-supported-brightgreen?logo=php&logoColor=white) | **Supported** | ASAN/UBSan-instrumented build; full driver & mutation |
| ![CPython](https://img.shields.io/badge/CPython-supported-brightgreen?logo=python&logoColor=white) | **Supported** | ASAN-instrumented build; full driver & mutation |
| ![Swift](https://img.shields.io/badge/Swift-supported-brightgreen?logo=swift&logoColor=white) | **Supported** | Official nightly build; driver and parser functional; ASan not yet enabled |
| ![MLIR](https://img.shields.io/badge/MLIR-in%20development-blue?logo=llvm&logoColor=white) | **In Development** | Official nightly build; driver and parser functional |
| ![Rust](https://img.shields.io/badge/Rust-supported-brightgreen?logo=rust&logoColor=white) | **Supported** | Driver and parser functional; may have edge-case bugs |
| ![Go](https://img.shields.io/badge/Go-experimental-orange?logo=go&logoColor=white) | **Experimental** | Parser and setup only; driver in progress |
| ![Lean](https://img.shields.io/badge/Lean-experimental-orange) | **Experimental** | Driver functional; limited seed corpus |
| ![WGSL](https://img.shields.io/badge/WGSL-experimental-orange) | **Experimental** | Naga/wgslc driver; limited testing |
| ![JavaScript](https://img.shields.io/badge/JavaScript-planned-lightgrey?logo=javascript&logoColor=white) | **Planned** | Not yet integrated |
| ![SQL](https://img.shields.io/badge/SQL-planned-lightgrey?logo=sqlite&logoColor=white) | **Planned** | Parser stub only |

## Bugs Found

Bugs found by Fusion Fuzz are tracked at https://fusion-fuzz.github.io (updated periodically).

## Setup & Usage

### Prerequisites

```bash
apt install -y git-lfs docker.io
pip install -r requirements.txt
```

### Running the Fuzzer

All targets run inside Docker to protect host integrity. The pattern is the same for every project:

```bash
# 1. Build the Docker image for your target
cd ./projects/<name>
docker build -t fusion-fuzz-<name> .
cd ../..

# 2. Start a container with the repo mounted
docker run --name fuzz-<name> -dit -m 24g \
  -v .:/home/fuzz/WorkSpace/fusion-fuzz \
  fusion-fuzz-<name>:latest

# 3. Enter the container and start fuzzing
docker exec -it fuzz-<name> bash
cd /home/fuzz/WorkSpace/fusion-fuzz
python3 main.py --project <name> --setup --bug-corpus
```

> **Memory limit:** Set `-m` to a value appropriate for your machine (e.g. `-m 16g`). This prevents OOM crashes caused by unbounded fuzzing programs.

Project-specific notes are below.

---

#### PHP

```bash
cd ./projects/php && docker build -t fusion-fuzz-php . && cd ../..
docker run --name fuzz-php -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-php:latest
docker exec -it fuzz-php bash -c "cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project php --setup --bug-corpus"
```

#### CPython

```bash
cd ./projects/cpython && docker build -t fusion-fuzz-cpython . && cd ../..
docker run --name fuzz-cpython -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-cpython:latest
docker exec -it fuzz-cpython bash -c "cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project cpython --setup --bug-corpus"
```

#### Swift

```bash
cd ./projects/swift && docker build -t fusion-fuzz-swift . && cd ../..
docker run --name fuzz-swift -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-swift:latest
docker exec -it fuzz-swift bash -c "cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project swift --setup --bug-corpus"
```

#### MLIR

```bash
cd ./projects/mlir && docker build -t fusion-fuzz-mlir . && cd ../..
docker run --name fuzz-mlir -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-mlir:latest
docker exec -it fuzz-mlir bash -c "cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project mlir --setup --bug-corpus"
```

> **Note:** The first MLIR setup compiles LLVM/MLIR from source, which can take several hours and requires substantial RAM. On a 32 GB machine, limit compilation parallelism (the Dockerfile does this automatically with `-j4`).

### CLI Reference

```
python3 main.py --project <name> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--project <name>` | *(required)* | Target project (folder name under `projects/`) |
| `--iterations <n>` | `-1` (unlimited) | Stop after N fuzzing iterations |
| `--setup` | off | Re-parse seeds and rebuild the corpus |
| `--bug-corpus` | off | Seed corpus with pre-translated reproducers from `corpus/corpus.db` |
| `--preprocessing` | off | Run dynamic info collection on seeds before fuzzing |
| `--dry-run` | off | Execute every seed once; discard seeds with non-zero exit codes |
| `--concurrency <n>` | from config | Override the number of parallel worker threads |
| `--sample-log [path]` | off | Log each sample's seeds and stdout/stderr (default path: `output/<project>_samples.log`) |
| `--reduce <bug_dir>` | — | Standalone: minimize a crash reproducer via delta debugging |
| `--signature <sig>` | — | Override the crash signature string used by `--reduce` |

## Output Structure

```text
output/
├── bugs/
│   └── <project>/
│       └── crash_<id>.md      # Metadata, logs, and reproduction content
└── <project>.db               # SQLite corpus DB
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

Fusion Fuzz is structured around five decoupled components:

### 1. Orchestrator (`core/orchestrator.py`)
The central fuzzing loop. It manages a dynamic thread pool, monitors for stalled workers, deduplicates crashes by signature (e.g., from AddressSanitizer output), and runs each iteration in an isolated temporary directory.

### 2. Drivers (`core/driver.py`, `projects/*/driver.py`)
Adapters that abstract target execution. Three layers:
- **`BaseDriver`** — CLI execution, signal analysis, and crash detection. All drivers inherit from this.
- **`DockerDriver`** *(extends `BaseDriver`)* — For targets running in persistent Docker containers. Handles container lifecycle, seed file transfer via a shared `.ffl_tmp/` volume mount, and the write→exec→cleanup→result loop. Subclasses only implement `_build_exec_cmd()`.
- **Project drivers** (e.g., `PHPDriver`, `LeanDriver`) — Override methods for target-specific logic (e.g., parsing `.phpt` headers, import hoisting, custom crash signatures).

### 3. Seed Parsers (`core/parser.py`, `projects/*/parser.py`)
Scan source trees, extract metadata, and populate a per-project `corpus.db` (SQLite). Two layers:
- **`BaseParser`** — Directory scanning, table creation/deduplication, and corpus loading.
- **Project parsers** — Implement `parse_content()` for language-specific metadata (imports, functions, structs, etc.).

### 4. Fusion Engine (`core/fusion.py`)
Merges two seeds into a novel input using:
- **Generic dataflow interleaving** — Weaves variable dependencies between two programs.
- **Language-specific strategies** — e.g., PHP class/property instrumentation and API fuzzing.

### 5. Mutation Engine (`core/mutation.py`)
- **`BaseMutator`** — Generic mutations: arithmetic/logical operators, integer constants.
- **`PHPMutator`** — PHP-specific mutations: `PHP_INT_MAX`, magic constants, variable replacement.

### 6. LLM Generator (`core/llmgen.py`)
Optional component that uses LLMs to generate fresh test cases and inject them into the corpus as new seeds.

## Adding a New Project

Open an issue and we will look into adding support.
