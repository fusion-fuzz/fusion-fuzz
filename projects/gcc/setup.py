"""
projects/gcc/setup.py — fetch and build GCC for FusionFuzz.

Called by main.py as setup(project_root) on the first run (or with
--setup). Leaves a usable compiler at
projects/gcc/gcc-install/bin/{gcc,g++} plus a seed corpus for the parser.

Build configuration, and why
----------------------------
`--enable-checking=yes,extra,rtl` is the single most valuable flag here.
GCC's internal consistency checks are what turn a silently wrong tree or
RTL into an "internal compiler error: in <fn>, at <file>:<line>" — the bug
class this fuzzer is built to find. It is the direct counterpart of
LLVM_ENABLE_ASSERTIONS=ON in the clang adapter.

`--disable-bootstrap` builds the compiler once with the host GCC rather
than three times with itself. A bootstrap costs hours and buys nothing for
fuzzing: we want a compiler that reports its own inconsistencies, not one
that has been proven to miscompile itself identically twice.

Sanitizers (FFL_GCC_SANITIZE=1) build cc1/cc1plus themselves under
-fsanitize=address,undefined. That catches memory errors the internal
checks miss, but it is off by default and deliberately so: an instrumented
cc1 runs roughly 2-3x slower, and this fuzzer's throughput is already
bounded by compile time. Turn it on for a targeted campaign, not for
routine fuzzing. (GCC also offers --with-build-config=bootstrap-asan, but
that requires the full three-stage bootstrap this build skips.)

Seed corpus
-----------
Extracted from GCC's own testsuite in the cloned source tree. These are
the tests GCC's developers wrote to pin down its behaviour, so they
concentrate exactly the constructs a compiler is likely to get wrong:
torture cases for the optimisers, regression tests for past ICEs, and
deliberate abuse of every corner of C and C++.

Which directories are taken, and why, is in TESTSUITE_DIRS below. The
short version: every standalone C or C++ translation unit, for every
target architecture. Only the testsuites for front ends this build does
not enable are left out (Fortran, D, Go, Ada, Modula-2).
"""

import multiprocessing
import os
import re
import shutil
import subprocess

GCC_REPO = "https://github.com/gcc-mirror/gcc.git"
# A branch, not trunk: release branches are what real users hit, and an
# unbuildable trunk revision would strand the whole adapter.
GCC_BRANCH = os.environ.get("FFL_GCC_BRANCH", "master")


def _run(cmd_str, cwd=None):
    print(f"[run] {cmd_str[:200]}")
    subprocess.run(["bash", "-c", cmd_str], check=True, cwd=cwd)


def _jobs():
    # GCC's link steps are memory-hungry, and this build usually shares the
    # host with running fuzz containers, so leave headroom rather than
    # taking every core. FFL_GCC_JOBS overrides.
    env = os.environ.get("FFL_GCC_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


def _seed_dir(project_root):
    return os.path.join(project_root, "seeds")


# Which testsuite directories to take, and the language each one is
# written in. The directory is authoritative about the language in a way
# the file extension is not: g++.old-deja uses ".C", g++.dg mixes ".C"
# and ".cc", and c-c++-common holds files meant to compile as either.
#
# Left out on purpose:
#   gfortran/gdc/gm2/   other front ends; --enable-languages=c,c++.
#   ada/go.test/objc.dg
#   jit.dg              libgccjit harness code, not translation units.
#   gcc.test-framework  tests of the test framework itself.
TESTSUITE_DIRS = {
    "gcc.dg":          "c",     # C front end + middle end
    "g++.dg":          "cpp",   # C++ — where GCC's ICEs concentrate
    "gcc.c-torture":   "c",     # optimiser torture; almost no negative tests
    "c-c++-common":    "c",     # valid as either language
    "g++.old-deja":    "cpp",   # legacy C++ suite, all ".C"
    # Every architecture, not just i386. These do not compile on an x86-64
    # build, and that is the point: they are rejected inside GCC rather
    # than by the driver, and cross-target constructs stress the target
    # hooks and builtin tables that assume they only ever see their own
    # backend.
    "gcc.target":      "c",
    "g++.target":      "cpp",
    "gcc.misc-tests":  "c",
}

# A file with no function definition and no dg-do directive is almost
# always a fragment another test #includes, not a translation unit. On
# the current tree this drops ~3,000 of ~60,000 candidates.
_FUNC_DEF_RE = re.compile(r'^\s*(?:[\w\*\s]+)\s+\**\w+\s*\([^;]*\)\s*\{', re.M)

_MAX_SEED_BYTES = 256 * 1024


def _looks_standalone(content):
    return bool(_FUNC_DEF_RE.search(content)) or "dg-do" in content


def _collect_testsuite_seeds(project_root, src_root):
    """Copy GCC's testsuite into projects/gcc/seeds/ as fusion seeds.

    Two things are normalised on the way in:

      * Language. Taken from the directory (see TESTSUITE_DIRS), not the
        extension, and written back out as ".c" or ".cc". Relying on the
        extension would silently mislabel all 3,229 g++.old-deja tests,
        whose ".C" suffix collapses to ".c" the moment anything lowercases
        it — they would then be handed to `gcc` and rejected as C.

      * Name. The testsuite has many same-named files in different
        directories (pr12345.c appears repeatedly). Seed identifiers must
        be unique or they overwrite each other in corpus.db, so the
        relative path becomes part of the name.
    """
    seeds = _seed_dir(project_root)
    testsuite = os.path.join(src_root, "gcc", "testsuite")
    if not os.path.isdir(testsuite):
        print(f"Warning: no testsuite at {testsuite}; corpus will be empty.")
        return 0

    # Start clean: a stale seeds/ from an earlier run would otherwise mix
    # in files this configuration no longer wants.
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = 0
    for rel_dir, lang in TESTSUITE_DIRS.items():
        root = os.path.join(testsuite, rel_dir)
        if not os.path.isdir(root):
            print(f"  (skipping {rel_dir}: not present in this tree)")
            continue
        out_ext = ".cc" if lang == "cpp" else ".c"
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith((".c", ".cc", ".C", ".cpp", ".cxx")):
                    continue
                src = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(src) > _MAX_SEED_BYTES:
                        skipped += 1
                        continue
                    with open(src, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except OSError:
                    skipped += 1
                    continue
                if not _looks_standalone(content):
                    skipped += 1
                    continue
                rel = os.path.relpath(src, testsuite)
                flat = os.path.splitext(rel)[0].replace(os.sep, "_")
                try:
                    with open(os.path.join(seeds, flat + out_ext), "w",
                              encoding="utf-8") as f:
                        f.write(content)
                    n += 1
                except OSError:
                    skipped += 1
        print(f"  {rel_dir:20s} {lang:3s} -> {n} seeds")
        copied += n

    print(f"Collected {copied} testsuite seeds into {seeds} "
          f"({skipped} skipped: fragments, oversized or unreadable)")
    return copied


def _build_corpus(project_root):
    """Parse seeds/ into corpus.db via the project's own parser.

    Going through parser.py rather than writing the DB here is what
    attaches the variables/dataflows metadata; without those keys dataflow
    fusion silently degrades to a no-op (see projects/gcc/parser.py).
    """
    seeds = _seed_dir(project_root)
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    # The corpus is derived entirely from the testsuite, so a stale DB is
    # never something to merge into — collect_seeds appends, and leftovers
    # from a previous corpus would silently survive.
    if os.path.exists(db):
        os.remove(db)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_gcc_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "gcc-src")
    build_dir = os.path.join(project_root, "gcc-build")
    install_dir = os.path.join(project_root, "gcc-install")
    gcc_bin = os.path.join(install_dir, "bin", "gcc")
    gxx_bin = os.path.join(install_dir, "bin", "g++")

    if not (os.path.exists(gcc_bin) and os.path.exists(gxx_bin)):
        if not os.path.exists(os.path.join(src_root, "gcc", "c", "c-parser.cc")):
            print(f"Cloning GCC ({GCC_BRANCH}) — shallow, this is ~1.5 GB ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {GCC_BRANCH} {GCC_REPO} {src_root}")

        # GCC insists on an out-of-tree build directory.
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(install_dir, exist_ok=True)

        sanitize = os.environ.get("FFL_GCC_SANITIZE", "") == "1"
        san_flags = ""
        if sanitize:
            # Instrument the compiler itself. -fno-sanitize-recover makes a
            # UBSan finding abort rather than print and continue, so the
            # driver's crash patterns see it.
            san = ("-fsanitize=address,undefined -fno-sanitize-recover=undefined "
                   "-fno-omit-frame-pointer -g")
            san_flags = f'CFLAGS="-O1 {san}" CXXFLAGS="-O1 {san}" LDFLAGS="{san}"'
            print("FFL_GCC_SANITIZE=1: building cc1/cc1plus under ASan+UBSan "
                  "(expect a 2-3x slower compiler)")

        # --disable-bootstrap: one build with the host compiler, not three
        #   with itself. Hours saved, nothing lost for fuzzing.
        # --enable-checking=yes,extra,rtl: the internal consistency checks
        #   that turn a bad tree/RTL into a reportable ICE. The whole point.
        # --disable-multilib avoids needing every target's libc, but we keep
        #   the 32-bit headers installed in the image so -m32 still parses.
        configure = f"""
set -e
cd {build_dir}
{san_flags} {src_root}/configure \\
    --prefix={install_dir} \\
    --enable-languages=c,c++ \\
    --disable-bootstrap \\
    --enable-checking=yes,extra,rtl \\
    --disable-multilib \\
    --disable-libsanitizer \\
    --disable-nls \\
    --disable-werror
"""
        _run(configure)
        _run(f"cd {build_dir} && make -j{_jobs()}")
        _run(f"cd {build_dir} && make install")

        # Tell the driver how this compiler was built. It matters: an
        # ASan-instrumented cc1 reserves ~20TB of virtual address space for
        # its shadow map at startup, so the driver's `ulimit -v` memory cap
        # would stop it from launching at all. The driver reads this marker
        # and switches to ASan's own hard_rss_limit_mb instead.
        if sanitize:
            with open(os.path.join(install_dir, ".ffl_sanitized"), "w") as f:
                f.write("address,undefined\n")

    if not os.path.exists(gcc_bin):
        raise RuntimeError(f"GCC build failed: {gcc_bin} not found")

    print(subprocess.run([gcc_bin, "--version"], capture_output=True,
                         text=True).stdout.splitlines()[0])

    # ---- seeds -------------------------------------------------------
    _collect_testsuite_seeds(project_root, src_root)
    _build_corpus(project_root)

    print(f"GCC setup complete. gcc: {gcc_bin}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). GCC instrumented with gcov is only
    useful for measuring which of its own passes the corpus reaches; it is
    not needed for bug finding, so this reuses the normal build."""
    setup(project_root)
