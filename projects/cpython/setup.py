"""
projects/cpython/setup.py — fetch and build an instrumented CPython.

Called by main.py as setup(project_root). Leaves an interpreter at
projects/cpython/cpython/build/python plus a seed corpus from CPython's
own Lib/test.

Bug oracles
-----------
--with-pydebug          The main one. It compiles in `assert()` throughout
                        the interpreter, the debug object allocator (the
                        pad-byte and API-mismatch checks in
                        Objects/obmalloc.c), and refcount bookkeeping —
                        `_Py_NegativeRefcount`. Without it almost
                        everything projects/cpython/analyzer.py looks for
                        is compiled out.
--with-address-sanitizer
                        Catches the use-after-free and buffer overflow that
                        a refcount bug turns into.
--with-undefined-behavior-sanitizer
                        New here. CPython's C is full of pointer arithmetic
                        and integer casts, and UBSan is the only oracle
                        that sees a shift-out-of-range or misaligned load
                        that happens to produce a plausible value.
--enable-experimental-jit
                        The JIT is the newest and least-exercised code in
                        the interpreter, which is exactly what a fuzzer
                        should be pointed at.

Seed corpus
-----------
Two sources, and the split matters:

  cpython-fuzzing-corpus.zip — 32,298 files, each one CPython test method
      decomposed into a standalone script (the filename encodes where it
      came from: `test_sys__<hash>__test_sys__UnraisableHookTest__...`).
      These carry the bulk of the corpus.

  the cloned tree's `Lib`, excluding `Lib/test` — the standard library
      modules themselves. Ordinary, heavily-exercised Python that reaches
      interpreter paths the tests do not.

`Lib/test` is deliberately *not* taken: the decomposed corpus already
covers the same tests at a much finer grain, and whole test files are slow
in a way that costs more than they return. Measured on this build:

              median   p90     p95     >2s     >5s
    Lib/test   0.62s   4.56s   9.0s    15%     9%
    decomposed 0.59s   1.06s   1.33s   1.3%    0.5%

The medians match because both are dominated by the ASan interpreter's
~0.4s startup. The tail is the whole difference, and the tail is what
holds a worker at the driver's timeout while teaching nothing — timeouts
were the second-largest cause of invalid children before this.

The zip's provenance is a real weakness: it is committed rather than
regenerated, so nothing here re-derives it from a given CPython revision.
Its filenames do trace back to specific test methods, which is enough to
audit a seed by hand but not enough to rebuild the set.
"""

import ast
import multiprocessing
import os
import shutil
import subprocess

CPYTHON_REPO = "https://github.com/python/cpython.git"
CPYTHON_BRANCH = os.environ.get("FFL_CPYTHON_BRANCH", "main")


def _run(cmd_str, cwd=None):
    print(f"[run] {cmd_str[:200]}")
    subprocess.run(["bash", "-c", cmd_str], check=True, cwd=cwd)


def _jobs():
    env = os.environ.get("FFL_CPYTHON_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# Taken from the cloned tree. `Lib/test` is excluded via _SKIP_DIRS — see
# the module docstring for the timing that decided it.
CORPUS_DIRS = ["Lib", "Tools/scripts"]

# The decomposed corpus, unpacked from the committed zip.
BUNDLED_CORPUS_ZIP = "cpython-fuzzing-corpus.zip"
BUNDLED_CORPUS_DIR = os.path.join("cpython-fuzzing-corpus", "corpus")

_MAX_SEED_BYTES = 256 * 1024

# Subtrees that are not standalone Python: data files, deliberately broken
# syntax used as parser fixtures, and the test *support* framework, which
# does nothing when run as a script.
_SKIP_DIRS = ("__pycache__", "/data/", "/tokenizedata/", "/encoded_modules/",
              "/xmltestdata/", "/decimaltestdata/", "/subprocessdata/",
              "/support/", "/certdata/",
              # GUI toolkits: importing them needs a display, so every
              # execution on one fails identically and learns nothing.
              "/idlelib/", "/tkinter/", "/turtledemo/",
              # Vendored third-party code, not CPython.
              "/site-packages/", "/ensurepip/_bundled/",
              # Lib/test/crashers is CPython's own list of *known*
              # interpreter crashes. Its README: "This directory only
              # contains tests for outstanding bugs that cause the
              # interpreter to segfault." Fusing one and reporting the
              # segfault is rediscovering a documented bug — the same
              # reason the Rust adapter leaves tests/crashes out. Missing
              # this cost a reduction that converged straight back onto
              # Lib/test/crashers/underlying_dict.py.
              "/crashers/",
              # Covered at finer grain, and much faster, by the decomposed
              # corpus. See the module docstring's timing table.
              "/test/")


def _unpack_bundled_corpus(project_root):
    """Unpack the decomposed-test corpus if it has not been already."""
    out = os.path.join(project_root, "cpython-fuzzing-corpus")
    if os.path.isdir(os.path.join(project_root, BUNDLED_CORPUS_DIR)):
        return os.path.join(project_root, BUNDLED_CORPUS_DIR)
    zip_path = os.path.join(project_root, BUNDLED_CORPUS_ZIP)
    if not os.path.exists(zip_path):
        print(f"Warning: {zip_path} not found; corpus will be Lib only.")
        return None
    import zipfile
    print(f"Unpacking {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(project_root)
    return os.path.join(project_root, BUNDLED_CORPUS_DIR)


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = skipped_unparseable = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(src_root, rel)
        if not os.path.isdir(root):
            print(f"  (skipping {rel}: not present)")
            continue
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            norm = dirpath.replace(os.sep, "/") + "/"
            if any(s in norm for s in _SKIP_DIRS):
                continue
            for name in files:
                if not name.endswith(".py"):
                    continue
                # `__main__.py` is a package entry point: running it runs
                # the whole suite beneath it. Measured, these were 17 of 17
                # fused children that hit the driver's timeout — one file,
                # Lib/test/test_ctypes/__main__.py, accounted for 11 — and
                # each one holds a worker for the full timeout while
                # exercising the test runner rather than the interpreter.
                if name == "__main__.py":
                    skipped += 1
                    continue
                src = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(src) > _MAX_SEED_BYTES:
                        skipped += 1
                        continue
                    content = open(src, encoding="utf-8", errors="ignore").read()
                except OSError:
                    skipped += 1
                    continue
                if not content.strip():
                    skipped += 1
                    continue
                # The seed has to be parseable by the Python running the
                # *fusion*, not just by the interpreter under test. The
                # container's host is 3.12 while the built target is 3.16,
                # and CPython's own Lib is written against the newer one —
                # `lazy import`, t-strings, PEP 695 type parameters.
                #
                # A seed the host cannot parse still fuses, but every
                # ast-based step in the CPython strategy silently falls
                # back to a line heuristic that splits parenthesised
                # imports and compound statements in half, so the child is
                # reliably broken. Better to leave it out than to spend
                # executions on output that cannot work.
                try:
                    ast.parse(content)
                except (SyntaxError, ValueError, RecursionError):
                    skipped_unparseable += 1
                    continue
                # test_foo.py recurs across subdirectories; seed identifiers
                # must be unique or they overwrite each other in corpus.db.
                flat = os.path.relpath(src, src_root).replace(os.sep, "_")
                with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        print(f"  {rel:16s} -> {n} files")
        copied += n
    bundled = _unpack_bundled_corpus(project_root)
    if bundled and os.path.isdir(bundled):
        n = 0
        for name in sorted(os.listdir(bundled)):
            if not name.endswith(".py"):
                continue
            src = os.path.join(bundled, name)
            try:
                if os.path.getsize(src) > _MAX_SEED_BYTES:
                    skipped += 1
                    continue
                content = open(src, encoding="utf-8", errors="ignore").read()
            except OSError:
                skipped += 1
                continue
            if not content.strip():
                skipped += 1
                continue
            try:
                ast.parse(content)
            except (SyntaxError, ValueError, RecursionError):
                skipped_unparseable += 1
                continue
            with open(os.path.join(seeds, "decomp_" + name), "w",
                      encoding="utf-8") as f:
                f.write(content)
            n += 1
        print(f"  {'decomposed':16s} -> {n} files")
        copied += n

    print(f"Collected {copied} Python seeds into {seeds} "
          f"({skipped} skipped: data files, oversized or empty; "
          f"{skipped_unparseable} skipped: newer syntax than this host's Python)")
    return copied


def _build_corpus(project_root):
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)      # corpus is derived entirely from the checkout
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_cpython_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "cpython")
    build_dir = os.path.join(src_root, "build")
    python_bin = os.path.join(build_dir, "python")

    if not os.path.exists(python_bin):
        if not os.path.exists(os.path.join(src_root, "configure")):
            print(f"Cloning CPython ({CPYTHON_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            # Shallow: the previous version cloned the full history, which
            # is several GB and buys nothing here.
            _run(f"git clone --depth=1 --branch {CPYTHON_BRANCH} "
                 f"{CPYTHON_REPO} {src_root}")

        # UBSan is opt-in per environment: it is the newest of the three
        # sanitizers here and the most likely to fail to configure on an
        # older toolchain, so it can be turned off without editing this file.
        ubsan = ("" if os.environ.get("FFL_CPYTHON_NO_UBSAN") == "1"
                 else " --with-undefined-behavior-sanitizer")
        configure = f"""
set -e
mkdir -p {build_dir}
if [ ! -f {build_dir}/Makefile ]; then
    cd {build_dir} && ../configure \\
        --with-pydebug \\
        --enable-experimental-jit=yes \\
        --with-address-sanitizer{ubsan}
fi
make -C {build_dir} -j{_jobs()}
"""
        _run(configure)

    if not os.path.exists(python_bin):
        raise RuntimeError(f"CPython build failed: {python_bin} not found")
    _run(f"{python_bin} --version")

    # pip under ASan is slow and occasionally trips the leak detector; the
    # modules below are what Lib/test imports that trunk no longer ships.
    # A failure here is not fatal — most seeds do not need them.
    test_deps = ["xdrlib3", "telnetlib3", "pyasynchat", "legacy-cgi", "pytest"]
    try:
        _run(f"ASAN_OPTIONS=detect_leaks=0 {python_bin} -m ensurepip --upgrade && "
             f"ASAN_OPTIONS=detect_leaks=0 {python_bin} -m pip install --quiet "
             f"{' '.join(test_deps)}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not install test dependencies ({e}); "
              "seeds importing them will fail to run.")

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"CPython setup complete. python: {python_bin}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Reuses the normal build: CPython's
    own coverage tooling measures Python-level code, not the C interpreter
    this fuzzer is testing."""
    setup(project_root)
