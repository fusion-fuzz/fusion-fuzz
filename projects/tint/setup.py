"""
projects/tint/setup.py — fetch and build the tint WGSL compiler for FusionFuzz.

Called by main.py as setup(project_root). Leaves the tint executable at
projects/tint/dawn-src/dawn/out/fuzz/tint plus a seed corpus taken from
tint's own WGSL test inputs.

tint is Dawn's WGSL compiler — the same language naga handles, so this
adapter reuses naga's core support wholesale (the WGSL lexicon, live-var
config, metadata collector and the three fusion strategies), the way GCC
reuses clang's and SpiderMonkey reuses V8's. What is tint-specific is this
file, the driver, and the crash oracle.

The checkout
------------
Unlike V8, tint needs no gclient. Dawn's CMake build can fetch its own
dependencies with a plain Python script (DAWN_FETCH_DEPENDENCIES=ON), so a
shallow git clone plus that fetch is enough, and only the tint command-line
tool is built — not the whole of Dawn.

Bug oracles, and why these
--------------------------
tint's checks are compiled in, so this is where most of the oracle lives.

  DAWN_ALWAYS_ASSERT
      Turns on TINT_ASSERT everywhere, not just in debug builds. This is
      tint's assertion mechanism: a broken internal invariant becomes
      `<file>:<line> internal compiler error: TINT_ASSERT(expr)` on stderr
      followed by a trap, instead of silently wrong output.

  TINT_ENABLE_IR_VALIDATION_ASSERTS (on by default; kept on)
      Validates tint's intermediate representation. The IR is where a
      mistranslation or a malformed transform first becomes detectable, so
      this is the single highest-value check for a compiler fuzzer — the
      counterpart of GCC's --enable-checking and V8's verify_csa.

  DAWN_ENABLE_ASAN / DAWN_ENABLE_UBSAN (FFL_TINT_ASAN=1, default on)
      A memory-safety bug in a C++ compiler usually manifests as a
      use-after-free or an out-of-bounds read reached through a malformed
      program. ASan/UBSan turn that into a report at the moment it happens.

A debug build with assertions and IR validation is slower than release,
but tint compiles a small shader in milliseconds either way, so the trade
is entirely worth it.
"""

import multiprocessing
import os
import shutil
import subprocess

DAWN_REPO = os.environ.get("FFL_TINT_REPO", "https://github.com/google/dawn.git")
DAWN_BRANCH = os.environ.get("FFL_TINT_BRANCH", "main")
OUT_DIR = "out/fuzz"
TARGET = "tint_cmd_tint_cmd"


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:200]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_TINT_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 16))


def cmake_args():
    """The CMake configuration. See the module docstring for the reasoning."""
    asan = os.environ.get("FFL_TINT_ASAN", "1") == "1"
    args = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Debug",
        # Dawn fetches its own dependencies rather than needing depot_tools.
        "-DDAWN_FETCH_DEPENDENCIES=ON",

        # The oracles.
        "-DDAWN_ALWAYS_ASSERT=ON",
        "-DTINT_ENABLE_IR_VALIDATION_ASSERTS=ON",

        # No window system. tint is a WGSL-to-text compiler and never opens
        # a surface, but Dawn's CMake pulls in GLFW for its samples, and
        # GLFW's build hard-fails without wayland-scanner.
        "-DDAWN_USE_GLFW=OFF",
        "-DDAWN_USE_WAYLAND=OFF",
        "-DDAWN_USE_X11=OFF",

        # No GPU backends: they need Vulkan/GL headers and build nothing
        # this adapter runs.
        "-DDAWN_ENABLE_VULKAN=OFF",
        "-DDAWN_ENABLE_DESKTOP_GL=OFF",
        "-DDAWN_ENABLE_OPENGLES=OFF",
        "-DDAWN_ENABLE_NULL=OFF",

        # ...but the code writers must be forced back ON. Their defaults
        # are tied to the backend switches just turned off
        # (TINT_BUILD_SPV_WRITER defaults to DAWN_ENABLE_VULKAN,
        # TINT_BUILD_MSL_WRITER to DAWN_ENABLE_METAL, which is macOS-only
        # and therefore already off here). Leaving them at their defaults
        # would build a tint that rejects `--format spirv|msl|hlsl|glsl`,
        # and projects/tint/driver.py picks the output backend as its main
        # lever — every run drawing a missing writer would be wasted.
        "-DTINT_BUILD_CMD_TOOLS=ON",
        "-DTINT_BUILD_WGSL_READER=ON",
        "-DTINT_BUILD_WGSL_WRITER=ON",
        "-DTINT_BUILD_SPV_WRITER=ON",
        "-DTINT_BUILD_MSL_WRITER=ON",
        "-DTINT_BUILD_HLSL_WRITER=ON",
        "-DTINT_BUILD_GLSL_WRITER=ON",
        # Powers --validate: SPIR-V uses SPIRV-Tools' validator, compiled
        # in rather than shelled out to.
        "-DTINT_BUILD_GLSL_VALIDATOR=ON",

        "-DTINT_BUILD_TESTS=OFF",
        "-DDAWN_BUILD_SAMPLES=OFF",
        "-DTINT_BUILD_BENCHMARKS=OFF",
        # Symbolised stacks; without them an ASan report is bare addresses.
        "-DCMAKE_CXX_FLAGS=-g1",
    ]
    if asan:
        args += ["-DDAWN_ENABLE_ASAN=ON", "-DDAWN_ENABLE_UBSAN=ON"]
    return " ".join(args)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# test/tint is tint's own WGSL test suite. Only the *input* shaders are
# taken, and only the hand-written ones:
#
#   *.expected.*  are golden outputs (the SPIR-V/MSL/HLSL a test should
#                 produce), not WGSL inputs.
#   builtins/gen  is 17k template-generated files exercising one builtin
#                 overload each — enormous, near-identical, and it would
#                 swamp the corpus with a single construct.
#
# What is left is ~3k hand-written shaders: the regression tests under
# bug/, the language-feature tests under expressions/, statements/, types/
# and the rest. These are to tint what gcc/testsuite is to GCC.
CORPUS_ROOT = "test/tint"

_MAX_SEED_BYTES = 256 * 1024
_SKIP_DIRS = ("/gen/",)


def _collect_seeds(project_root, dawn_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    root = os.path.join(dawn_root, CORPUS_ROOT)
    copied = skipped = 0
    for dirpath, _dirs, files in os.walk(root):
        norm = dirpath.replace(os.sep, "/") + "/"
        if any(sk in norm for sk in _SKIP_DIRS):
            continue
        for name in files:
            if not name.endswith(".wgsl"):
                continue
            # Golden outputs, not inputs.
            if ".expected." in name:
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
            # 947.wgsl recurs across bug/dawn, bug/chromium, ...; identifiers
            # must be unique or they overwrite each other in corpus.db.
            flat = os.path.relpath(src, root).replace(os.sep, "_")
            with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                f.write(content)
            copied += 1
    print(f"Collected {copied} WGSL seeds into {seeds} "
          f"({skipped} skipped: oversized or empty)")
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
        "ffl_tint_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def tint_path(project_root):
    return os.path.join(project_root, "dawn-src", "dawn", OUT_DIR, "tint")


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "dawn-src")
    dawn_root = os.path.join(src_root, "dawn")
    tint = tint_path(project_root)
    env = dict(os.environ)

    if not os.path.exists(tint):
        if not os.path.isdir(os.path.join(dawn_root, "src", "tint")):
            print("Cloning Dawn (shallow; blobs fetched on demand).")
            os.makedirs(src_root, exist_ok=True)
            _run(f"git clone --depth=1 --filter=blob:none "
                 f"-b {DAWN_BRANCH} {DAWN_REPO} {dawn_root}", env=env)

        args = cmake_args()
        print(f"CMake args: {args}")
        _run(f"cd {dawn_root} && cmake -B {OUT_DIR} {args}", env=env)
        # The fetch of Dawn's dependencies happens during configure; the
        # build then only needs the one target.
        _run(f"cd {dawn_root} && ninja -C {OUT_DIR} -j{_jobs()} {TARGET}", env=env)

    if not os.path.exists(tint):
        raise RuntimeError(f"tint build failed: {tint} not found")

    # tint has no --version flag (only --help), so the build is identified
    # by the Dawn commit it came from.
    rev = subprocess.run(["git", "-C", dawn_root, "rev-parse", "HEAD"],
                         capture_output=True, text=True, errors="replace")
    print(f"tint built from Dawn {(rev.stdout or '').strip() or 'unknown'}")

    _collect_seeds(project_root, dawn_root)
    _build_corpus(project_root)
    print(f"tint setup complete. binary: {tint}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Coverage instrumentation here would
    measure the WGSL under test, not the compiler, so it adds nothing."""
    setup(project_root)
