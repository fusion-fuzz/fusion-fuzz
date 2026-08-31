"""
projects/spidermonkey/setup.py — fetch and build the SpiderMonkey shell.

Called by main.py as setup(project_root). Leaves a shell at
projects/spidermonkey/sm-src/firefox/obj/dist/bin/js plus a seed corpus
taken from SpiderMonkey's own JIT test suite.

The checkout
------------
Unlike V8, SpiderMonkey needs no gclient: the JS engine has a standalone
autoconf build under js/src, so a shallow git clone of the Firefox tree is
enough.

The checkout is deliberately *not* sparse. A sparse cone covering the
obvious directories (js, build, python, config, mfbt, ...) gets as far as
configure and then fails on whatever it did not think of — `.cargo/
config.toml.in`, referenced from the tree's root moz.build, was the one
that surfaced here. The build reads scattered files across the tree and
enumerating them is guesswork that fails late; `--filter=blob:none` keeps
the fetch small enough that a full checkout is the cheaper trade.

Bug oracles, and why these
--------------------------
  --enable-debug
      The whole point. It turns on MOZ_ASSERT, which is SpiderMonkey's
      assertion mechanism and the direct counterpart of V8's DCHECK: a
      broken internal invariant becomes
      `[pid] Assertion failure: <expr>, at <file>:<line>` on stderr
      instead of silently wrong JIT output. Nearly every interesting
      finding in this engine comes through it.

  --disable-optimize
      Keeps the assertions honest and the stacks readable. A debug build
      that is also optimised inlines the frame that would name the culprit.

  --enable-oom-breakpoint
      Makes the shell's artificial-OOM machinery available, which is what
      the `oomTest` used by parts of the corpus needs.

  ASan (FFL_SM_ASAN=1, default on)
      A JIT bug in a dynamically typed language usually manifests as a
      type confusion that reads freed or wrong memory. ASan is what turns
      that into a report at the moment it happens rather than a crash
      somewhere unrelated later.

Unlike V8, the tier controls here are *runtime* flags (--ion-eager and
friends), not build options, so they live in driver.py.
"""

import multiprocessing
import os
import shutil
import subprocess

SM_REPO = os.environ.get(
    "FFL_SM_REPO", "https://github.com/mozilla-firefox/firefox.git")
SM_BRANCH = os.environ.get("FFL_SM_BRANCH", "main")
OBJ_DIR = "obj"




def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:200]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_SM_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 16))


def _build_env():
    env = dict(os.environ)
    cargo = os.path.expanduser("~/.cargo/bin")
    if os.path.isdir(cargo) and cargo not in env.get("PATH", ""):
        env["PATH"] = f"{cargo}:{env['PATH']}"
    # js/src/configure was generated with autoconf 2.13 and rejects newer.
    env.setdefault("AUTOCONF", "autoconf2.13")
    # Mozilla's build system phones home unless told not to.
    env.setdefault("MOZ_NOSPAM", "1")
    env.setdefault("SHELL", "/bin/bash")
    return env


def configure_args():
    """The configure flags. See the module docstring for the reasoning."""
    args = [
        "--enable-debug",
        "--disable-optimize",
        "--enable-oom-breakpoint",
        # The shell only; no browser, no JIT-less special casing.
        "--disable-jemalloc",
        "--disable-tests",
        "--without-intl-api",
    ]
    if os.environ.get("FFL_SM_ASAN", "1") == "1":
        args += ["--enable-address-sanitizer", "--disable-jemalloc"]
    return " ".join(dict.fromkeys(args))


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# js/src/jit-test/tests is SpiderMonkey's JIT regression suite: ~9,300
# standalone scripts, each pinning down one behaviour. It is to this engine
# what mjsunit is to V8 and gcc/testsuite is to GCC.
#
# js/src/tests is deliberately left out: it is ~60,000 files, but it is
# mostly an imported copy of test262 driven by a harness (shell.js/browser.js
# loaded alongside each test), so the files are not standalone.
CORPUS_DIRS = ["js/src/jit-test/tests"]

_MAX_SEED_BYTES = 256 * 1024

# Directories whose files are helpers rather than tests, or that need a
# companion file the corpus does not carry.
_SKIP_DIRS = ("/lib/", "/asm.js/", "/wasm/", "/modules/", "/debug/")


def _collect_seeds(project_root, sm_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    copied = skipped = 0
    for rel in CORPUS_DIRS:
        root = os.path.join(sm_root, rel)
        if not os.path.isdir(root):
            print(f"  (skipping {rel}: not present)")
            continue
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            norm = dirpath.replace(os.sep, "/") + "/"
            if any(sk in norm for sk in _SKIP_DIRS):
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
                # bug1234.js recurs across directories; identifiers must be
                # unique or they overwrite each other in corpus.db.
                flat = os.path.relpath(src, sm_root).replace(os.sep, "_")
                with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        print(f"  {rel:28s} -> {n} files")
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
        "ffl_sm_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def js_shell_path(project_root):
    return os.path.join(project_root, "sm-src", "firefox", "js", "src",
                        OBJ_DIR, "dist", "bin", "js")


def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "sm-src")
    sm_root = os.path.join(src_root, "firefox")
    js_src = os.path.join(sm_root, "js", "src")
    shell = js_shell_path(project_root)
    env = _build_env()

    if not os.path.exists(shell):
        if not os.path.isdir(os.path.join(js_src, "shell")):
            print("Cloning SpiderMonkey — the Firefox tree is large; "
                  "blobs are fetched on demand.")
            os.makedirs(src_root, exist_ok=True)
            _run(f"git clone --depth=1 --filter=blob:none "
                 f"-b {SM_BRANCH} {SM_REPO} {sm_root}", env=env)

        args = configure_args()
        print(f"configure args: {args}")
        os.makedirs(os.path.join(js_src, OBJ_DIR), exist_ok=True)
        _run(f"cd {js_src}/{OBJ_DIR} && ../configure {args}", env=env)
        _run(f"cd {js_src}/{OBJ_DIR} && make -j{_jobs()}", env=env)

    if not os.path.exists(shell):
        raise RuntimeError(f"SpiderMonkey build failed: {shell} not found")

    ver = subprocess.run([shell, "--version"], capture_output=True, text=True,
                         errors="replace")
    print((ver.stdout or ver.stderr).strip() or "js shell built")

    _collect_seeds(project_root, sm_root)
    _build_corpus(project_root)
    print(f"SpiderMonkey setup complete. shell: {shell}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Coverage instrumentation here would
    measure the JavaScript under test, not the engine, so it adds nothing."""
    setup(project_root)
