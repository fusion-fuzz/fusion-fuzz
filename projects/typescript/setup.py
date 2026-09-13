"""
projects/typescript/setup.py — build the Go-native TypeScript compiler
and lift a seed corpus out of its own test suite.

Called by main.py as setup(project_root). Leaves the compiler at
projects/typescript/tsc-bin and seeds in projects/typescript/seeds.

Build
-----
microsoft/TypeScript master is the Go port (TypeScript 7): `tsc/` is a
Go module, `go build ./cmd/tsc` is the whole build (lib.d.ts files are
embedded). `-race` is optional (FFL_TSC_RACE=1): the checker runs in
parallel by default (`--checkers`), so a data race in it is a real bug,
but a race-instrumented binary is several times slower.

Seed corpus
-----------
tsc/testdata/tests/cases/{compiler,conformance}: ~12,500 test cases.
The single-file ones (~9,800) are taken whole, `// @option: value`
directive lines included — projects/typescript/driver.py turns them into
command-line options. Multi-file cases (`// @filename:`) are left out:
their halves import each other.
"""

import multiprocessing
import os
import re
import shutil
import subprocess

TS_REPO = "https://github.com/microsoft/TypeScript.git"
TS_BRANCH = os.environ.get("FFL_TS_BRANCH", "main")

CORPUS_DIRS = ["tsc/testdata/tests/cases/compiler", "tsc/testdata/tests/cases/conformance"]
_MAX_SEED_BYTES = 32 * 1024
_FILENAME_RE = re.compile(r'^\s*//\s*@[Ff]ile[Nn]ame\s*:', re.M)


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)
    n = skipped = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(src_root, rel)
        label = os.path.basename(rel)
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(".ts") or name.endswith(".d.ts"):
                    continue
                p = os.path.join(dirpath, name)
                if os.path.getsize(p) > _MAX_SEED_BYTES:
                    skipped += 1
                    continue
                with open(p, encoding="utf-8", errors="replace") as f:
                    text = f.read()
                if _FILENAME_RE.search(text) or not text.strip():
                    skipped += 1
                    continue
                sub = os.path.relpath(dirpath, root).replace(os.sep, "_")
                out = f"{label}_{sub}_{name}" if sub != "." else f"{label}_{name}"
                with open(os.path.join(seeds, out), "w", encoding="utf-8") as f:
                    f.write(text)
                n += 1
    print(f"Collected {n} seeds into {seeds} ({skipped} skipped: multi-file or too large)")
    return n


def _build_corpus(project_root):
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_typescript_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "typescript-src")
    tsc_bin = os.path.join(project_root, "tsc-bin")

    if not os.path.exists(tsc_bin):
        if not os.path.exists(os.path.join(src_root, "tsc", "go.mod")):
            print(f"Cloning TypeScript ({TS_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {TS_BRANCH} {TS_REPO} {src_root}")
        env = dict(os.environ)
        env["GOPROXY"] = env.get("GOPROXY") or "https://proxy.golang.org,direct"
        if env["GOPROXY"] == "off":
            env["GOPROXY"] = "https://proxy.golang.org,direct"
        env.pop("GOFLAGS", None)
        race = " -race" if os.environ.get("FFL_TSC_RACE") == "1" else ""
        _run(f"cd {src_root}/tsc && go build{race} -o {tsc_bin} ./cmd/tsc", env=env)

    if not os.path.exists(tsc_bin):
        raise RuntimeError(f"tsc build failed: {tsc_bin} not found")
    print(subprocess.run([tsc_bin, "--version"], capture_output=True, text=True).stdout.strip())

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"TypeScript setup complete. tsc: {tsc_bin}")


def setup_cov(project_root):
    setup(project_root)


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
