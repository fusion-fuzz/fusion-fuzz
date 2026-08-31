"""
projects/v8/setup.py — fetch and build d8 (V8's shell) for FusionFuzz.

Called by main.py as setup(project_root). Leaves a shell at
projects/v8/v8-src/v8/out/fuzz/d8 plus a seed corpus from V8's own
mjsunit suite.

The checkout
------------
V8 cannot be used from a plain `git clone`. The build needs a dozen
sibling repositories — build/, buildtools/, third_party/icu, and the
toolchain V8 pins — and only `gclient` knows where each one goes. So this
runs depot_tools' `fetch v8`, which is slower and much larger than a clone
(~12 GB) but is the only checkout the build accepts.

`--no-history` keeps it to a shallow fetch; V8's full history is several
times the working tree and nothing here reads it.

Bug oracles, and why these
--------------------------
V8's checks are compiled in, so this is where most of the oracle lives —
unlike rustc or Go, where the interesting checks are per-invocation flags.

  is_debug + v8_enable_slow_dchecks
      DCHECKs, including the expensive ones. This is V8's assertion
      mechanism and the direct counterpart of LLVM_ENABLE_ASSERTIONS: a
      broken internal invariant becomes a reported failure instead of
      silently wrong JIT output.

  v8_enable_verify_heap
      Walks the heap and checks every object's shape. Catches a GC or
      allocation bug at the point it corrupts something rather than
      wherever the corruption is later read.

  v8_enable_verify_csa
      Verifies CodeStubAssembler output — the hand-written machine-level
      code behind builtins, and a place where a mistake produces working
      code that is subtly wrong.

  v8_enable_object_print / v8_code_comments
      Not checks, but they make a crash report legible: without them a
      DCHECK failure names an address instead of an object.

  is_asan (FFL_V8_ASAN=1, default on)
      A JIT bug usually manifests as a type confusion that reads freed or
      wrong memory. ASan is what turns that into a report at the moment it
      happens.

A debug build with slow DCHECKs and heap verification is roughly an order
of magnitude slower than release. That is the right trade here: the point
is to see the failure, and V8 executes a fused program in milliseconds
either way.
"""

import multiprocessing
import os
import shutil
import subprocess

V8_FETCH_TARGET = "v8"
V8_BRANCH = os.environ.get("FFL_V8_BRANCH", "main")
OUT_DIR = "out/fuzz"


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_V8_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    # V8's link steps are memory-hungry; leave headroom.
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


def _depot_tools_env():
    env = dict(os.environ)
    depot = env.get("DEPOT_TOOLS_PATH", "/opt/depot_tools")
    if os.path.isdir(depot) and depot not in env.get("PATH", ""):
        env["PATH"] = f"{depot}:{env['PATH']}"
    env.setdefault("DEPOT_TOOLS_METRICS", "0")
    # Skip depot_tools' self-update. It runs before every gclient/fetch
    # invocation and hangs indefinitely against the depth=1 clone the
    # Dockerfile makes — no git child, no CPU, no timeout. The clone is
    # already at HEAD, so there is nothing to update to.
    env.setdefault("DEPOT_TOOLS_UPDATE", "0")
    # depot_tools ships its own Python bootstrap that expects Google's
    # internal environment; this tells it to use the system interpreter.
    env.setdefault("VPYTHON_BYPASS",
                   "manually managed python not supported by chrome operations")
    return env


def gn_args():
    """The GN argument string. See the module docstring for the reasoning."""
    asan = os.environ.get("FFL_V8_ASAN", "1") == "1"
    args = [
        "is_debug = true",
        "target_cpu = \"x64\"",
        # The assertions. `slow_dchecks` adds the ones V8 considers too
        # expensive for its own debug bots — exactly the ones worth having
        # when throughput is not the binding constraint.
        "v8_enable_slow_dchecks = true",
        "v8_enable_verify_heap = true",
        "v8_enable_verify_csa = true",
        # Make a failure readable rather than a bare address.
        "v8_enable_object_print = true",
        "v8_code_comments = true",
        # d8 is the shell this adapter runs; the rest of the targets are
        # not built.
        "v8_monolithic = false",
        "v8_use_external_startup_data = false",
        # Symbolised stacks; without them an ASan report is addresses.
        "symbol_level = 1",
        "use_goma = false",
    ]
    if asan:
        args += ["is_asan = true", "is_lsan = false"]
    return " ".join(args)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# mjsunit is V8's own JavaScript regression suite: ~9,200 files, each a
# standalone script written to pin down one behaviour. It is to V8 what
# gcc/testsuite is to GCC.
#
# Left out:
#   test/wasm-spec-tests, test/wasm-js  — generated, enormous, and mostly
#       .wasm binaries rather than JS.
#   test/inspector, test/debugger       — driven by a protocol harness, not
#       runnable as plain scripts.
#   test/intl                           — needs a full ICU data set.
CORPUS_DIRS = ["test/mjsunit", "test/message"]

_MAX_SEED_BYTES = 256 * 1024

# mjsunit files that are not standalone: they are `load()`ed by others, or
# they drive a worker/module that d8 cannot run from a single file.
_SKIP_DIRS = ("/tools/", "/wasm-js/", "/regress/wasm/")


def _collect_seeds(project_root, v8_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(v8_root, rel)
        if not os.path.isdir(root):
            print(f"  (skipping {rel}: not present)")
            continue
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            norm = dirpath.replace(os.sep, "/") + "/"
            if any(s in norm for s in _SKIP_DIRS):
                continue
            for name in files:
                if not name.endswith(".js"):
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
                # regress-12345.js recurs across directories; identifiers
                # must be unique or they overwrite each other in corpus.db.
                flat = os.path.relpath(src, v8_root).replace(os.sep, "_")
                with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        print(f"  {rel:18s} -> {n} files")
        copied += n
    print(f"Collected {copied} JavaScript seeds into {seeds} "
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
        "ffl_v8_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def d8_path(project_root):
    return os.path.join(project_root, "v8-src", "v8", OUT_DIR, "d8")


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "v8-src")
    v8_root = os.path.join(src_root, "v8")
    d8 = d8_path(project_root)
    env = _depot_tools_env()

    if not os.path.exists(d8):
        if not os.path.isdir(os.path.join(v8_root, "src")):
            print("Fetching V8 with gclient — this is ~12 GB and slow.")
            os.makedirs(src_root, exist_ok=True)
            # `fetch` writes .gclient into the *current* directory and puts
            # the checkout in a v8/ subdirectory beneath it.
            _run(f"cd {src_root} && fetch --no-history {V8_FETCH_TARGET}", env=env)
        _run(f"cd {v8_root} && git checkout {V8_BRANCH} 2>/dev/null || true", env=env)
        # From src_root, not v8_root: `fetch` writes .gclient into the
        # directory it was run in, and gclient resolves the checkout
        # relative to that file. Running sync from inside v8/ makes it walk
        # up to find the same file and warn that .gclient_entries is
        # missing, because the entries are bookkeeping for src_root.
        _run(f"cd {src_root} && gclient sync --no-history -D", env=env)

        args = gn_args()
        print(f"GN args: {args}")
        _run(f"cd {v8_root} && gn gen {OUT_DIR} --args='{args}'", env=env)
        _run(f"cd {v8_root} && autoninja -j{_jobs()} -C {OUT_DIR} d8", env=env)

    if not os.path.exists(d8):
        raise RuntimeError(f"V8 build failed: {d8} not found")

    ver = subprocess.run([d8, "--version"], capture_output=True, text=True,
                         errors="replace")
    print((ver.stdout or ver.stderr).strip().splitlines()[0]
          if (ver.stdout or ver.stderr) else "d8 built")

    _collect_seeds(project_root, v8_root)
    _build_corpus(project_root)
    print(f"V8 setup complete. d8: {d8}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). V8's coverage instrumentation measures
    the JavaScript under test, not the engine, so it adds nothing here."""
    setup(project_root)
