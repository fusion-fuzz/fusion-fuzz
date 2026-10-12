"""
projects/lean/setup.py — build the Lean 4 compiler under test from source
and lift a seed corpus out of its own test suite.

Called by main.py as setup(project_root) inside the fuzz-lean container.
Leaves the toolchains under projects/lean/lean4/build/<variant>/stage1,
their description in projects/lean/toolchain.json, and seeds under
projects/lean/seeds.

Build
-----
leanprover/lean4 master bootstraps from the C sources checked in under
stage0/ (stage0 is always a plain release build), and stage1 — the
compiler that is fuzzed — is configured per variant:

  relwithassert  CMAKE_BUILD_TYPE=RelWithAssert, the repo's own preset:
                 `-O3 -DLEAN_DEBUG` for the C++ kernel and runtime *and*,
                 through LEANC_OPTS, for the C that the Lean-written
                 frontend (elaborator, tactics, LCNF/IR compiler) is
                 compiled to. Every `lean_assert` in the kernel type
                 checker, the runtime object model and lean.h's inline
                 helpers is live; a violation prints
                 "LEAN ASSERTION VIOLATION" and aborts.
  asan           RelWithAssert plus the repo's `sanitize` preset flags:
                 -fsanitize=address,undefined on the C++ side and on the
                 stage1 C, mimalloc off (ASan needs to see the
                 allocations). Built second and drawn by the driver for a
                 minority of runs (it is 2-3x slower). Skip it with
                 FFL_LEAN_ASAN=0.

Both are built with clang so that ccache can share stage0 between them.
FFL_LEAN_JOBS sets the parallelism (default: all cores). Do not build
while the fuzzer runs: its stale-process reaper (core/driver.py
cleanup_stale_processes) SIGKILLs any `lean` older than a few timeouts
whose command line names projects/lean — which includes the build's own
long-running stage1 compilations. A variant whose
stage1/bin/lean already exists is not rebuilt; FFL_LEAN_REBUILD=1 forces
a rebuild (incremental).

Seed corpus
-----------
The test suite's single-file piles (tests/README.md): `elab` (elaborate
and compare messages), `elab_fail` (expected to exit 1), `compile`
(programs with `main`, run compiled and interpreted), plus `elabissues`,
`lean`, `simpperf`, `playground` and the two `*_bench` piles. A file is
skipped when the harness gives it an `.init.sh`/`.before.sh` (extra
arguments or files the fuzzer would not supply), when it imports anything
but Init/Std/Lean (a sibling module or a Lake package), or when it is
larger than _MAX_SEED_BYTES.

`#guard_msgs` is the suite's assertion: the docstring above it is the
expected output of the command it wraps. A fused program changes values
on purpose, so an `info:`-only expectation (an `#eval`'s printed value)
would turn every real dataflow edge into a mismatch error. Those guards
and their docstrings are removed — the command still runs and prints.
Guards that expect an `error:` or `warning:` are kept: removing them
would make the seed fail on its own, and a mismatch still prints the
actual messages, which is all the oracle needs.

`elab_fail` seeds get a first-line marker (`-- ffl: expect-fail`) so
config.yaml's drop_seeds_failing_alone_except keeps them: their error
recovery paths are the point.
"""

import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys

LEAN_REPO = "https://github.com/leanprover/lean4.git"
LEAN_BRANCH = os.environ.get("FFL_LEAN_BRANCH", "master")

_MAX_SEED_BYTES = 48 * 1024

# (test pile, seed group). Order is the order seeds are written.
SEED_PILES = [
    ("elab", "elab"),
    ("elab_fail", "elab_fail"),
    ("compile", "compile"),
    ("elabissues", "elabissues"),
    ("lean", "lean"),
    ("simpperf", "simpperf"),
    ("playground", "playground"),
    ("elab_bench", "bench"),
    ("compile_bench", "bench"),
]

EXPECT_FAIL_MARKER = "-- ffl: expect-fail"

# Text the oracle (projects/lean/analyzer.py) treats as a bug report. A
# test that *expects* one — a kernel rejection provoked through `addDecl`,
# a deliberate `panic!`, an `unreachable!` — would have its children
# reproduce it and be filed as a finding, so such a test is not a seed.
# Checked against the test and against its `.out.expected`.
_ORACLE_TEXT_RE = re.compile(
    r'\(kernel\)|PANIC|LEAN ASSERTION|internal error|Internal error|'
    r'compiler IR check failed|\(interpreter\)|unreachable code|'
    r'invalid LCNF|LCNF (?:let|local|parameter)|out of scope')

_IMPORT_RE = re.compile(r'^\s*(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?([\w.«»]+)', re.M)
_ALLOWED_IMPORT_ROOTS = ("Init", "Std", "Lean", "Lake")


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:240]}", flush=True)
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_LEAN_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, multiprocessing.cpu_count())


# ---------------------------------------------------------------------------
# Toolchain
# ---------------------------------------------------------------------------

def _ensure_repo(project_root):
    src = os.path.join(project_root, "lean4")
    if not os.path.isfile(os.path.join(src, "CMakePresets.json")):
        print("Cloning leanprover/lean4 (shallow)...", flush=True)
        shutil.rmtree(src, ignore_errors=True)
        _run(f"git clone --depth=1 --branch {LEAN_BRANCH} {LEAN_REPO} {src}")
    return src


# cmake cache variables per variant. Stage0 discards LEAN_EXTRA_CXX_FLAGS
# and friends (it is a plain release build by design), so these only
# reach stage1 — the compiler under test.
_SAN_CXX = "-g1 -fsanitize=address,undefined -fno-sanitize-recover=undefined " \
           "-DLEAN_DEFAULT_THREAD_STACK_SIZE=16*1024*1024"
_SAN_CC = "-g1 -fsanitize=address,undefined -fno-sanitize-recover=undefined"
_SAN_LD = "-fsanitize=address,undefined -fsanitize-link-c++-runtime -fuse-ld=lld"

VARIANTS = {
    "relwithassert": {
        "CMAKE_BUILD_TYPE": "RelWithAssert",
        "STRIP_BINARIES": "OFF",
        "WFAIL": "OFF",
    },
    "asan": {
        "CMAKE_BUILD_TYPE": "RelWithAssert",
        "LEAN_EXTRA_CXX_FLAGS": _SAN_CXX,
        "LEANC_EXTRA_CC_FLAGS": _SAN_CC,
        "LEAN_EXTRA_LINKER_FLAGS": _SAN_LD,
        "USE_MIMALLOC": "OFF",
        "BSYMBOLIC": "OFF",
        "STRIP_BINARIES": "OFF",
        "WFAIL": "OFF",
    },
}

# What a sanitized lean needs at run time. Leaks are not findings here:
# the frontend deliberately never frees the environment at exit.
SAN_ENV = {
    "ASAN_OPTIONS": "detect_leaks=0:abort_on_error=1:symbolize=1:"
                    "allocator_may_return_null=1:detect_stack_use_after_return=0",
    "UBSAN_OPTIONS": "print_stacktrace=1:halt_on_error=1:symbolize=1",
}


def _stage1(src, variant):
    return os.path.join(src, "build", variant, "stage1")


def build_variant(src, variant):
    """Configure and build one stage1 toolchain. Incremental: re-running
    after an interrupted build resumes it."""
    bdir = os.path.join(src, "build", variant)
    flags = " ".join(f"-D{k}='{v}'" for k, v in VARIANTS[variant].items())
    env = dict(os.environ, CC="clang", CXX="clang++")
    env.setdefault("CCACHE_DIR", os.path.expanduser("~/.ccache"))
    if not os.path.exists(os.path.join(bdir, "CMakeCache.txt")):
        _run(f"cmake -S . -B {bdir} -G 'Unix Makefiles' "
             f"-DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ {flags}",
             cwd=src, env=env)
    _run(f"make -C {bdir} -j{_jobs()} stage1", cwd=src, env=env)


def _toolchain_entry(src, variant):
    s1 = _stage1(src, variant)
    lean = os.path.join(s1, "bin", "lean")
    if not os.path.exists(lean):
        return None
    leanc = os.path.join(s1, "bin", "leanc")
    entry = {
        "variant": variant,
        "lean": lean,
        "leanc": leanc if os.path.exists(leanc) else None,
        "libdir": os.path.join(s1, "lib", "lean"),
        "env": dict(SAN_ENV) if variant == "asan" else {},
    }
    return entry


def _smoke(entry):
    """Elaborate one small program that imports Lean, runs the compiler
    and the kernel, and prints. Returns (ok, seconds)."""
    import time
    prog = ("import Lean\nopen Lean Meta\n"
            "def fib : Nat → Nat | 0 => 0 | 1 => 1 | n+2 => fib n + fib (n+1)\n"
            "theorem fib_pos : fib 5 = 5 := by decide\n"
            "#eval fib 20\n"
            "#eval show MetaM Unit from do logInfo m!\"{← inferType (mkConst ``Nat.add)}\"\n")
    tmp = os.path.join(os.path.dirname(entry["libdir"]), "..", "ffl_smoke.lean")
    tmp = os.path.abspath(tmp)
    with open(tmp, "w") as f:
        f.write(prog)
    env = dict(os.environ, **entry["env"])
    t = time.time()
    r = subprocess.run([entry["lean"], tmp], capture_output=True, text=True, env=env,
                       stdin=subprocess.DEVNULL, timeout=600)
    dt = time.time() - t
    os.remove(tmp)
    ok = r.returncode == 0 and "6765" in r.stdout
    print(f"  smoke [{entry['variant']}]: rc={r.returncode} {dt:.1f}s "
          f"{'ok' if ok else (r.stdout + r.stderr)[-400:]}", flush=True)
    return ok, dt


def setup_toolchains(project_root, src):
    want = ["relwithassert"]
    if os.environ.get("FFL_LEAN_ASAN", "1") != "0":
        want.append("asan")
    toolchains = []
    for variant in want:
        lean = os.path.join(_stage1(src, variant), "bin", "lean")
        if not os.path.exists(lean) or os.environ.get("FFL_LEAN_REBUILD") == "1":
            try:
                build_variant(src, variant)
            except subprocess.CalledProcessError as e:
                print(f"Warning: building the {variant} toolchain failed ({e}); "
                      "continuing without it.", flush=True)
                continue
        entry = _toolchain_entry(src, variant)
        if entry is None:
            continue
        ok, dt = _smoke(entry)
        if ok:
            entry["smoke_seconds"] = round(dt, 2)
            toolchains.append(entry)
    if not toolchains:
        print("Error: no Lean toolchain could be built.", flush=True)
        sys.exit(1)
    info = {"lean_src": src, "toolchains": toolchains}
    try:
        info["githash"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src,
                                         capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    with open(os.path.join(project_root, "toolchain.json"), "w") as f:
        json.dump(info, f, indent=2)
    for t in toolchains:
        print(f"  toolchain: {t['variant']}: {t['lean']}")
    return info


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# A `#guard_msgs` header: optional spec in parentheses, then `in`. The
# wrapped command may follow on the same line or on the next.
_GUARD_RE = re.compile(r'^(?P<indent>[ \t]*)#guard_msgs\b(?:\s*\([^)\n]*\))?\s+in\b[ \t]*', re.M)
# The expected-output docstring immediately above a guard.
_DOC_BEFORE_RE = re.compile(r'/--(?P<body>(?:(?!-/)[\s\S])*?)-/\s*$')


def strip_info_guards(content):
    """Remove `#guard_msgs` guards (with their docstrings) whose
    expectation is informational only. See the module docstring."""
    out = []
    pos = 0
    for m in _GUARD_RE.finditer(content):
        before = content[pos:m.start()]
        doc = _DOC_BEFORE_RE.search(before)
        expected = doc.group("body") if doc else ""
        if re.search(r'^\s*(?:error|warning)\b', expected, re.M):
            continue                     # keep: the command is expected to fail
        if doc:
            before = before[:doc.start()]
        out.append(before)
        out.append(m.group("indent"))
        pos = m.end()
    out.append(content[pos:])
    text = "".join(out)
    # A guard whose command was on the next line leaves a blank `in` line
    # behind only if the regex stopped at end of line; collapse runs of
    # blank lines it created.
    return re.sub(r'\n{3,}', '\n\n', text)


def _imports_ok(content):
    for mod in _IMPORT_RE.findall(content):
        root = mod.split(".", 1)[0].strip("«»")
        if root not in _ALLOWED_IMPORT_ROOTS:
            return False
    return True


def _collect_seeds(src, seeds_dir):
    tests = os.path.join(src, "tests")
    counts = {}
    for pile, group in SEED_PILES:
        base = os.path.join(tests, pile)
        if not os.path.isdir(base):
            continue
        dst_dir = os.path.join(seeds_dir, group)
        os.makedirs(dst_dir, exist_ok=True)
        n = 0
        for root, dirs, files in os.walk(base):
            # A `<name>.lean.dir` holds a test's data files, not tests.
            dirs[:] = [d for d in dirs if not d.endswith(".dir")]
            for fname in sorted(files):
                if not fname.endswith(".lean"):
                    continue
                path = os.path.join(root, fname)
                if any(os.path.exists(path + ext) for ext in
                       (".init.sh", ".before.sh", ".no_test", ".dir", ".input")):
                    continue
                try:
                    if os.path.getsize(path) > _MAX_SEED_BYTES:
                        continue
                    with open(path, encoding="utf-8") as f:
                        content = f.read()
                except (OSError, UnicodeDecodeError):
                    continue
                if not _imports_ok(content) or re.search(r'^prelude\b', content, re.M):
                    continue
                expected = ""
                if os.path.exists(path + ".out.expected"):
                    try:
                        with open(path + ".out.expected", encoding="utf-8", errors="replace") as f:
                            expected = f.read()
                    except OSError:
                        pass
                if _ORACLE_TEXT_RE.search(content) or _ORACLE_TEXT_RE.search(expected):
                    continue
                # `#exit` stops elaboration: everything a fusion appends
                # after it would never be checked.
                m = re.search(r'^#exit\b', content, re.M)
                if m:
                    content = content[:m.start()]
                content = strip_info_guards(content)
                if pile == "elab_fail":
                    content = f"{EXPECT_FAIL_MARKER}\n{content}"
                rel = os.path.relpath(path, base).replace(os.sep, "__")
                with open(os.path.join(dst_dir, f"{pile}__{rel}"), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        counts[pile] = n
    return counts


def setup(project_root):
    project_root = os.path.abspath(project_root)
    print(f"Setting up Lean 4 in: {project_root}", flush=True)
    src = _ensure_repo(project_root)

    seeds_dir = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds_dir, ignore_errors=True)
    os.makedirs(seeds_dir, exist_ok=True)
    counts = _collect_seeds(src, seeds_dir)
    print("  seeds:", ", ".join(f"{k}={v}" for k, v in counts.items()), flush=True)

    # corpus.db is rebuilt from the seed tree: a stale row for a seed whose
    # normalisation changed would otherwise shadow the new text.
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)

    setup_toolchains(project_root, src)
    print("Lean setup complete.", flush=True)


if __name__ == "__main__":
    # `python3 projects/lean/setup.py build <variant>` builds one toolchain
    # without touching the seeds (used to start the long build early).
    root = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) >= 3 and sys.argv[1] == "build":
        build_variant(_ensure_repo(root), sys.argv[2])
    else:
        setup(root)
