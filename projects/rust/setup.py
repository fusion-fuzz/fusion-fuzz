"""
projects/rust/setup.py — fetch and build rustc for FusionFuzz.

Called by main.py as setup(project_root). Leaves a compiler at
projects/rust/rust-src/build/<triple>/stage1/bin/rustc plus a seed corpus
from rust-lang/rust's own test suite.

This replaced a setup that used whatever `rustc` was on PATH (the image
was rustlang/rust:nightly). A released nightly is built with neither of
the two assertion sets this fuzzer depends on, so most of what it looks
for was compiled out.

Bug oracles, and why these
--------------------------
rust.debug-assertions  — rustc's own `debug_assert!`s. The direct
                         counterpart of LLVM_ENABLE_ASSERTIONS for clang:
                         without them a broken internal invariant becomes
                         wrong output instead of an ICE.

llvm.assertions        — assertions inside the LLVM rustc links against.
                         bootstrap.example.toml is explicit about why this
                         matters: "When assertions are disabled, bugs in
                         the integration between rustc and LLVM can lead to
                         unsoundness (segfaults, etc.) in the rustc process
                         itself, not just in the generated code."

The second one would normally mean building LLVM from source — hours. It
does not here: Rust's CI publishes an assertions-enabled LLVM to a
separate artifact server (`artifacts_with_llvm_assertions_server` in
src/stage0, the "-alt" builds), and bootstrap downloads from it when
`llvm.assertions = true` is combined with `download-ci-llvm`. So this
build gets LLVM assertions at the cost of a download.

overflow-checks        — arithmetic overflow in rustc itself panics rather
                         than wrapping silently.

Stage 1, not stage 2: a stage1 rustc is a full compiler built by the
downloaded stage0 with our flags, so it carries the assertions. Stage 2
would only add "built by a compiler that also has them", which changes
nothing about the binary's own checking and roughly doubles the build.

Seed corpus
-----------
rust-lang/rust's tests/ui and friends — 21k files whose whole purpose is
to pin down compiler behaviour. tests/crashes is deliberately excluded:
it is a directory of *known* ICE reproducers, and including it would bury
anything new under 206 already-filed bugs.
"""

import multiprocessing
import os
import platform
import shutil
import subprocess

RUST_REPO = "https://github.com/rust-lang/rust.git"
# rust-lang/rust's default branch is `main`, not `master` — cloning with
# `--branch master` fails outright ("Remote branch master not found").
RUST_BRANCH = os.environ.get("FFL_RUST_BRANCH", "main")


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_RUST_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


def host_triple():
    machine = {"x86_64": "x86_64", "aarch64": "aarch64"}.get(platform.machine(),
                                                             platform.machine())
    return f"{machine}-unknown-linux-gnu"


def _write_bootstrap_toml(src_root):
    """Write bootstrap.toml with every oracle we can afford.

    `download-ci-llvm = true` together with `assertions = true` is the
    combination that matters: bootstrap then fetches from the
    assertions-enabled artifact server rather than building LLVM. If the
    host triple has no such artifact, bootstrap falls back to building
    LLVM from source — slower, same result.
    """
    path = os.path.join(src_root, "bootstrap.toml")
    if os.path.exists(path):
        print(f"Keeping existing {path}")
        return
    config = f"""# Written by projects/rust/setup.py — see the module docstring.
profile = "compiler"
change-id = "ignore"

[llvm]
# The reason this adapter builds rustc rather than using a nightly.
assertions = true
download-ci-llvm = true
# NOTE: do not add `optimize` here. Bootstrap rejects it outright when
# combined with download-ci-llvm ("Setting `llvm.optimize` is incompatible
# with `llvm.download-ci-llvm`") — the downloaded artifact's own build
# settings are fixed, so any llvm.* option other than `assertions` is a
# configuration error rather than a preference.

[build]
target = ["{host_triple()}"]
docs = false
extended = false
# Stage1 is a complete compiler carrying our assertions; see the module
# docstring for why stage2 buys nothing here.
build-stage = 1
test-stage = 1

[rust]
# rustc's own debug_assert!s — the point of building from source.
debug-assertions = true
# The standard library is compiled input here, not the thing under test,
# and instrumenting it slows every compilation for no extra coverage of
# the compiler.
debug-assertions-std = false
overflow-checks = true
# Line tables only: full debuginfo makes the build much larger and slower,
# but without any we get no symbol names in an ICE backtrace, which is
# most of a crash signature.
debuginfo-level = 1
codegen-units = 0
incremental = false
# Keeps `RUST_BACKTRACE=1` on an ICE from being a bare address dump.
backtrace = true
"""
    with open(path, "w") as f:
        f.write(config)
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# Test trees worth taking. tests/ui alone is 21k files and is where the
# language's edge cases live.
#
# Left out:
#   tests/crashes    206 *known* ICE reproducers. Including them would mean
#                    every run rediscovers 206 filed bugs before finding
#                    anything new.
#   tests/run-make   driver scripts, not translation units.
#   tests/rustdoc*   exercise rustdoc, a different binary.
#   tests/debuginfo  assert on debugger output, not on compilation.
CORPUS_DIRS = ["ui", "mir-opt", "codegen-units", "pretty", "assembly-llvm",
               "incremental", "coverage"]

_MAX_SEED_BYTES = 256 * 1024

# compiletest headers that mean the file is not a standalone compilation:
# it needs a companion crate, a specific target, or an external tool.
_NOT_STANDALONE = (
    "//@ aux-build", "//@ aux-crate", "//@ proc-macro",
    "//@ ignore-test", "//@ needs-", "//@ only-",
    "// aux-build", "// ignore-test",
)


def _is_standalone(content):
    head = "\n".join(content.splitlines()[:60])
    return not any(marker in head for marker in _NOT_STANDALONE)


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    tests_root = os.path.join(src_root, "tests")
    if not os.path.isdir(tests_root):
        print(f"Warning: no tests tree at {tests_root}; corpus will be empty.")
        return 0

    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(tests_root, rel)
        if not os.path.isdir(root):
            print(f"  (skipping {rel}: not present)")
            continue
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            # auxiliary/ holds the companion crates aux-build tests need;
            # they are never compiled on their own.
            if os.sep + "auxiliary" in dirpath:
                continue
            for name in files:
                if not name.endswith(".rs"):
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
                if not _is_standalone(content):
                    skipped += 1
                    continue
                # issue-NNNNN.rs recurs across directories; seed identifiers
                # must be unique or they overwrite each other in corpus.db.
                flat = os.path.relpath(src, tests_root).replace(os.sep, "_")
                with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        print(f"  {rel:16s} -> {n} files")
        copied += n
    print(f"Collected {copied} Rust seeds into {seeds} "
          f"({skipped} skipped: aux-dependent, oversized or unreadable)")
    return copied


def _build_corpus(project_root):
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)      # corpus is derived entirely from the test tree
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_rust_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def rustc_path(project_root):
    return os.path.join(project_root, "rust-src", "build", host_triple(),
                        "stage1", "bin", "rustc")


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "rust-src")
    rustc = rustc_path(project_root)

    if not os.path.exists(rustc):
        if not os.path.exists(os.path.join(src_root, "x.py")):
            print(f"Cloning rust-lang/rust ({RUST_BRANCH}) — ~500 MB ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {RUST_BRANCH} {RUST_REPO} {src_root}")

        _write_bootstrap_toml(src_root)
        # `library` as well as the compiler: without a stage1 std, the
        # stage1 rustc cannot compile anything that uses the prelude, which
        # is every seed.
        _run(f"cd {src_root} && python3 x.py build --stage 1 "
             f"compiler/rustc library --jobs {_jobs()}")

    if not os.path.exists(rustc):
        raise RuntimeError(f"Rust build failed: {rustc} not found")

    ver = subprocess.run([rustc, "--version", "--verbose"],
                         capture_output=True, text=True)
    print(ver.stdout.strip())

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"Rust setup complete. rustc: {rustc}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Rust's -C instrument-coverage
    instruments the program under test, not the compiler, so it adds
    nothing here."""
    setup(project_root)
