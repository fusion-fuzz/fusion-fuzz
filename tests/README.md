# FusionFuzzLoop Tests

Run from the **project root**:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -t . -v
```

No Docker, no compilers, no fuzzer needed — all tests use mocks or in-memory SQLite.

## What's covered

| File | What's tested | Mocking needed? |
|------|---------------|-----------------|
| `core/test_driver.py` | `BaseDriver` and `DockerDriver`: `_run_command`, `_check_crash`, `execute`, `_ensure_container` | `subprocess.Popen/run` mocked |
| `core/test_parser.py` | `BaseParser`: `_save_to_db`, `load_corpus`, `collect_seeds`, dedup, schema | Real temp dirs + SQLite |
| `projects/test_parsers.py` | `parse_content()` for rust, lean, naga/wgsl, swift, gcc, mlir | None (pure functions) |
| `projects/test_signatures.py` | `extract_crash_signature()` for cpython, go, lean, php, clang, swift | None (pure functions) |

## What's NOT covered (intentionally)

- Actual Docker execution or container management
- Real compiler invocations
- The fuzzing loop (`core/orchestrator.py`)
- LLM generation (`core/llmgen.py`)
- Semantic fusion logic (`core/fusion.py`) — candidates for future tests
