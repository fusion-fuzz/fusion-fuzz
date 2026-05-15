# Fusion Fuzz

![Tests](https://github.com/fusion-fuzz/fusion-fuzz/actions/workflows/tests.yml/badge.svg)

**Fusion Fuzz** is a scalable fusion-based fuzzer designed to uncover deep bugs with rich semantics in language processors (compiler and interpreter) such as Rust and Python.

Unlike traditional grammar-based fuzzers (which generate code from scratch) or mutation-based fuzzers (which indiscriminately flip bits), fusion-fuzz operates on a higher semantic level. It generates high-quality fuzzing inputs by **fusing** two or more valid, pre-existing unit tests (seeds) via dataflow interleaving. And some other new features will be coming soon.

## Project Support

| Project | Status | Notes |
|---------|--------|-------|
| ![PHP](https://img.shields.io/badge/PHP-supported-brightgreen?logo=php&logoColor=white) | **Supported** | ASAN/UBSan-instrumented build; full driver & mutation |
| ![CPython](https://img.shields.io/badge/CPython-supported-brightgreen?logo=python&logoColor=white) | **Supported** | ASAN-instrumented build; full driver & mutation |
| ![Swift](https://img.shields.io/badge/Swift-supported-brightgreen?logo=swift&logoColor=white) | **Supported** | Official Nightly Build; Driver and parser functional; TODO: Enable ASan |
| ![Rust](https://img.shields.io/badge/Rust-supported-brightgreen?logo=rust&logoColor=white) | **Supported** | Official Nightly Build; Driver and parser functional |
| ![MLIR](https://img.shields.io/badge/MLIR-in%20development-yellow?logo=llvm&logoColor=white) | **In Development** | Driver and parser functional; may have bugs |
| ![Go](https://img.shields.io/badge/Go-experimental-orange?logo=go&logoColor=white) | **Experimental** | Parser and setup only; driver in progress |
| ![Lean](https://img.shields.io/badge/Lean-experimental-orange) | **Experimental** | Driver functional; limited seed corpus |
| ![WGSL](https://img.shields.io/badge/WGSL-experimental-orange) | **Experimental** | Naga/wgslc driver; limited testing |
| ![JavaScript](https://img.shields.io/badge/JavaScript-planned-lightgrey?logo=javascript&logoColor=white) | **Planned** | Not yet integrated |
| ![SQL](https://img.shields.io/badge/SQL-planned-lightgrey?logo=sqlite&logoColor=white) | **Planned** | Parser stub only |

## Bugs Found

Bugs found by fusion-fuzz are available at https://fusion-fuzz.github.io. This webpage will be updated from time to time.

## Setup & Usage

### Prerequisites

**Dependencies:** `apt install -y git-lfs docker.io`

### Running the Fuzzer

To ensure the host integrity, we strongly suggest to run fusion-fuzz inside docker. The docker file can be found in ./projects/<name>/Dockerfile. 

---

**For PHP:**

In host:
```bash
# 1. Build fuzzing docker
cd ./projects/php
docker build -t fusion-fuzz-php .
cd ../..
# 2. Start docker and mount fusion fuzz
docker run --name fuzz-php -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-php:latest
# strongly suggest to add the memory limit according to your specs, e.g., 16GB in the above command, to prevent OOB cases incurred by random fuzzing programs
```

Then go to the docker:
```bash
docker exec -it fusion-fuzz-php bash
cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project php --setup --bug-corpus
```

---

**For CPython:**

In host:
```bash
# 1. Build fuzzing docker
cd ./projects/cpython
docker build -t fusion-fuzz-cpython .
cd ../..
# 2. Start docker and mount fusion fuzz
docker run --name fuzz-cpython -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-cpython:latest
```

Then go to the docker:
```bash
cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project cpython --setup --bug-corpus
```

---

**For Swift:**

In host:
```bash
# 1. Build fuzzing docker
cd ./projects/swift
docker build -t fusion-fuzz-swift . # this may cost some time
cd ../..
# 2. Start docker and mount fusion fuzz
docker run --name fuzz-swift -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-swift:latest
```

Then go to the docker:
```bash
cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project swift --setup --bug-corpus
```

---

**For MLIR:**

In host:
```bash
# 1. Build fuzzing docker
cd ./projects/mlir
docker build -t fusion-fuzz-mlir . # this may cost some time
cd ../..
# 2. Start docker and mount fusion fuzz
docker run --name fuzz-mlir -dit -m 24g -v .:/home/fuzz/WorkSpace/fusion-fuzz fusion-fuzz-mlir:latest
```

Then go to the docker:
```bash
cd /home/fuzz/WorkSpace/fusion-fuzz && python3 main.py --project mlir --setup --bug-corpus
```

(first setup MLIR could take a long time..(a few hours) it can easily go OOM when compilation in personal computer, e.g., 32G RAM; thus only run it with 4 jobs)



## Output Structure

```text
output/
├── bugs/
│   └── <project_name>/
│       └── crash_<id>.md      # Markdown report: metadata, logs, and reproduction content
└── <project_name>.db          # SQLite corpus DB
```

Unique crashes are saved as **Markdown files** containing:
- **Metadata:** Exit code, execution duration, crash signature.
- **Logs:** Complete `STDOUT` and `STDERR`.
- **Reproduction:** The exact fused content that triggered the crash.

## Core Architecture (co-designed with Gemini and Claude)

FFL operates on a modular architecture designed to decouple the fuzzing logic from the target execution environment:

### 1. Orchestrator (`core/orchestrator.py`)
The central brain of the framework. Manages the lifecycle of the fuzzing loop:
- **Parallel Execution:** Schedules tasks across a dynamic thread pool to maximize CPU utilization.
- **Health Monitoring:** Automatically detects stalled threads and restarts the worker pool to ensure continuous operation.
- **Crash Deduplication:** Intelligent signature extraction (e.g., from AddressSanitizer output) to avoid reporting duplicate bugs.
- **Workspace Isolation:** Runs every iteration in a temporary directory to prevent file system pollution.

### 2. Drivers (`core/driver.py`, `projects/*/driver.py`)
Adapters that abstract away the target's execution complexity. Three layers:

- **`BaseDriver`** — Handles standard CLI execution, signal analysis, and crash detection. All drivers inherit from this.
- **`DockerDriver`** *(extends `BaseDriver`)* — Base for any target that runs inside a persistent Docker container. Manages container lifecycle (`_ensure_container`), seed file transfer via the shared `.ffl_tmp/` volume mount, and the write→exec→cleanup→result loop. Subclasses only need to implement `_build_exec_cmd()`. Currently used by: `clang`, `mlir`, `naga`, `rust`, `wgslc`.
- **Project Drivers** (e.g., `PHPDriver`, `LeanDriver`) — Override specific methods for project-specific logic (e.g., parsing `.phpt` headers, import hoisting, custom crash signatures).

### 3. Seed Parsers (`core/parser.py`, `projects/*/parser.py`)
Scan language-specific source trees, extract metadata, and store seeds in a per-project `corpus.db`. Two layers:

- **`BaseParser`** — Handles directory scanning, SQLite table creation/deduplication, and corpus loading. Subclasses set `extensions` and `seed_type`, and override `parse_content()`.
- **Project Parsers** (e.g., `RustParser`, `LeanParser`) — Implement `parse_content()` for language-specific metadata extraction (imports, functions, structs, etc.).

### 4. Fusion Engine (`core/fusion.py`)
Implements strategies to merge two seeds into a novel input:
- **Generic Dataflow:** Interleaves variable dependencies between two programs.
- **Language-Specific Strategies:** E.g., PHP class/property instrumentation and API fuzzing.

### 5. Mutation Engine (`core/mutation.py`)
- **`BaseMutator`:** Generic mutations (arithmetic, logical operators, integers).
- **`PHPMutator`:** Language-specific mutations (PHP_INT_MAX, magic constants, variable replacement).

### 6. LLM Generator (`core/llmgen.py`)
Optional component that injects fresh "genetic material" into the corpus using Large Language Models to generate new test cases on the fly.

## Adding a New Project

We have covered most compilers and interpeters. If you want supports for new ones, please create an issue.
