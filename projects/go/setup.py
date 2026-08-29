"""
projects/go/setup.py — fetch and build the Go toolchain for FusionFuzz.

Called by main.py as setup(project_root). Leaves a working toolchain at
projects/go/go-src/bin/go plus a seed corpus from Go's own test suite.

Bug oracles, and why these
--------------------------
Go's compiler has no configure-time "enable checking" switch the way GCC
does; its checks are runtime flags passed per compilation, so the work of
turning them on lives in projects/go/driver.py rather than here. The one
that matters most is

    -gcflags=-d=ssa/check/on      "enables checking after each phase"
                                  (src/cmd/compile/internal/ssacompile/compile.go)

which verifies SSA form after every optimisation pass. It is the direct
counterpart of GCC's --enable-checking=rtl: without it a pass that
produces malformed SSA usually yields silently wrong code instead of a
reportable crash.

What *this* file can do is build a compiler that is itself instrumented:

  FFL_GO_RACE=1   rebuilds cmd/compile with the race detector. The Go
                  compiler is concurrent (it compiles functions in
                  parallel, -c=N), so a data race in it is a real bug and
                  one that no amount of per-compilation flags will find.
                  Off by default: a race-instrumented compiler is roughly
                  2-5x slower and this fuzzer is compile-bound.

A plain `make.bash` build already gives us the rest. Go ships assertions
(base.Assert / base.Fatalf) in the released compiler rather than
compiling them out, so unlike LLVM there is no assertions-vs-release
distinction to configure — every build reports "internal compiler error".

Seed corpus
-----------
Go's test/ tree: ~3,400 files written to pin down compiler behaviour,
including a large body of regression tests for past ICEs. Files are taken
whole, their leading `// run` / `// compile` / `// errorcheck` directive
included, because parser.py reads it.
"""

import multiprocessing
import os
import shutil
import subprocess

GO_REPO = "https://github.com/golang/go.git"
GO_BRANCH = os.environ.get("FFL_GO_BRANCH", "master")


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_GO_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


def _bootstrap_root():
    """Where to find the Go that builds Go.

    src/make.bash needs GOROOT_BOOTSTRAP pointing at a toolchain at least
    as new as its `bootgo` line (1.26.0 at time of writing). The Docker
    image installs one at /usr/local/go-bootstrap and exports the variable;
    this falls back to the usual locations so the adapter still works
    outside that image.
    """
    env = os.environ.get("GOROOT_BOOTSTRAP")
    if env and os.path.exists(os.path.join(env, "bin", "go")):
        return env
    home = os.path.expanduser("~")
    for cand in ("/usr/local/go-bootstrap", "/usr/local/go",
                 os.path.join(home, "sdk", "go1.26.0"),
                 os.path.join(home, "go1.26.0")):
        if os.path.exists(os.path.join(cand, "bin", "go")):
            return cand
    return None


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

# Go's test/ tree is flat-ish; these are the subtrees worth taking beyond
# the top level. Directories whose tests are only meaningful as a *set*
# (the compiler's own dir-based tests) are left out — a single file from
# them does not compile alone.
CORPUS_SUBDIRS = ["fixedbugs", "typeparam", "escape", "inline", "codegen",
                  "abi", "loopvar", "intrinsic"]

# Directives that mean "this file is not a standalone compilation unit".
# rundir/compiledir/errorcheckdir tests are whole directories; a file from
# one is a fragment. Skipped tests are skipped for a reason.
_NOT_STANDALONE = ("rundir", "compiledir", "errorcheckdir", "runindir", "skip")

_MAX_SEED_BYTES = 256 * 1024


def _directive_of(content):
    """The leading action directive of a Go test file, if any.

    Go's test files start with a comment naming what the test harness
    should do: `// run`, `// compile`, `// errorcheck`, `// build`. It is
    the closest thing Go has to a DejaGnu directive, and it tells us
    whether the file is expected to compile at all.
    """
    for line in content.splitlines()[:20]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("//"):
            word = line[2:].strip().split(" ")[0].split("\t")[0]
            if word and word[0].isalpha():
                return word
            continue
        # Any non-comment line means the header is over.
        if not line.startswith("/*"):
            break
    return None


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    test_root = os.path.join(src_root, "test")
    if not os.path.isdir(test_root):
        print(f"Warning: no test tree at {test_root}; corpus will be empty.")
        return 0

    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)

    roots = [("test", test_root)]
    for sub in CORPUS_SUBDIRS:
        p = os.path.join(test_root, sub)
        if os.path.isdir(p):
            roots.append((f"test_{sub}", p))

    copied = skipped = 0
    seen = set()
    for label, root in roots:
        n = 0
        for dirpath, _dirs, files in os.walk(root):
            for name in files:
                if not name.endswith(".go"):
                    continue
                src = os.path.join(dirpath, name)
                if src in seen:
                    continue
                seen.add(src)
                try:
                    if os.path.getsize(src) > _MAX_SEED_BYTES:
                        skipped += 1
                        continue
                    content = open(src, encoding="utf-8", errors="ignore").read()
                except OSError:
                    skipped += 1
                    continue
                if _directive_of(content) in _NOT_STANDALONE:
                    skipped += 1
                    continue
                # Same-named files recur across subdirectories (issue*.go
                # especially); identifiers must be unique in corpus.db.
                flat = os.path.relpath(src, test_root).replace(os.sep, "_")
                with open(os.path.join(seeds, flat), "w", encoding="utf-8") as f:
                    f.write(content)
                n += 1
        print(f"  {label:18s} -> {n} files")
        copied += n
    print(f"Collected {copied} Go seeds into {seeds} "
          f"({skipped} skipped: directory-tests, oversized or unreadable)")
    return copied


def _build_corpus(project_root):
    """Parse seeds/ into corpus.db through the project's own parser, which
    records each seed's package clause, imports and directive — all three
    are needed at fusion time."""
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)      # corpus is derived entirely from the test tree
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_go_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "go-src")
    go_bin = os.path.join(src_root, "bin", "go")

    if not os.path.exists(go_bin):
        if not os.path.exists(os.path.join(src_root, "src", "make.bash")):
            print(f"Cloning Go ({GO_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {GO_BRANCH} {GO_REPO} {src_root}")

        boot = _bootstrap_root()
        if not boot:
            raise RuntimeError(
                "No bootstrap Go found. Building Go trunk needs an existing "
                "toolchain >= the `bootgo` line in src/make.bash (1.26.0). "
                "Set GOROOT_BOOTSTRAP, or use projects/go/Dockerfile, which "
                "installs one at /usr/local/go-bootstrap.")
        print(f"Bootstrapping from {boot}")

        env = dict(os.environ)
        env["GOROOT_BOOTSTRAP"] = boot
        # make.bash runs the standard library tests by default via all.bash;
        # make.bash alone just builds, which is what we want.
        env["GOMAXPROCS"] = str(_jobs())
        _run(f"cd {os.path.join(src_root, 'src')} && ./make.bash", env=env)

    if not os.path.exists(go_bin):
        raise RuntimeError(f"Go build failed: {go_bin} not found")

    env = dict(os.environ)
    env["GOROOT"] = src_root
    print(subprocess.run([go_bin, "version"], capture_output=True, text=True,
                         env=env).stdout.strip())

    # Optional: rebuild the compiler under the race detector. The Go
    # compiler compiles functions in parallel, so this is the only way to
    # see a race in it — but it makes every compilation markedly slower.
    if os.environ.get("FFL_GO_RACE") == "1":
        print("FFL_GO_RACE=1: rebuilding cmd/compile with -race "
              "(expect a much slower compiler)")
        _run(f"{go_bin} install -race cmd/compile", env=env)

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"Go setup complete. go: {go_bin}")


def setup_cov(project_root):
    """Coverage build (--setup-cov). Go's own coverage tooling instruments
    the program under test, not the compiler, so it adds nothing here."""
    setup(project_root)
