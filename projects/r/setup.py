"""
projects/r/setup.py — build the R interpreter under test and lift a seed
corpus out of R's own documentation and test suite.

Called by main.py as setup(project_root). Leaves an instrumented
interpreter at projects/r/install/bin/R and seeds in projects/r/seeds.

Build
-----
r-devel (the R Core Team's development trunk, github.com/r-devel/r-svn),
configured with:

  * AddressSanitizer on the C and C++ sources (clang >= 18). The Fortran
    parts are built by gfortran without instrumentation — legal, they
    simply are not checked — because R's configure rejects a Fortran
    compiler it cannot link a plain test program with.
  * `--enable-strict-barrier`: R's write-barrier checks. R's garbage
    collector is the part a fused program is most likely to break, and
    without this the breakage is silent until something else trips over
    it. This is what R Core themselves use for their sanitizer builds.
  * `--with-recommended-packages=no`: the recommended packages are
    downloaded, and nothing here may reach the network mid-fuzz.
  * X11, Java, cairo and tcltk off: they add no interpreter surface and
    a display is not available.

Seed corpus
-----------
Three sources, all from the cloned tree:

  1. `\\examples{...}` blocks in src/library/*/man/*.Rd — 1168 of them,
     each a short self-contained program written to demonstrate one
     function. This is the R analogue of the decomposed CPython corpus:
     small, varied, and written by the people who wrote the interpreter.
     `\\dontrun{}` and `\\donttest{}` bodies are dropped (they are marked
     as not-to-be-run), `\\dontshow{}` is kept but unwrapped.
  2. tests/*.R — R's own regression tests.
  3. src/library/*/tests/*.R — the per-package tests.
"""

import multiprocessing
import os
import re
import shutil
import subprocess

R_REPO = "https://github.com/r-devel/r-svn"
R_BRANCH = os.environ.get("FFL_R_BRANCH", "master")

_MAX_SEED_BYTES = 24 * 1024


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_R_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


# ---------------------------------------------------------------------------
# Rd examples
# ---------------------------------------------------------------------------

def _matching_brace(text, start):
    """Index just past the `}` matching the `{` at `start`."""
    depth, i, n = 0, start, len(text)
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _strip_rd_markup(body):
    """An \\examples body as runnable R.

    `\\dontrun{}` and `\\donttest{}` mark code R itself does not run when
    checking a package — it needs a network, a display, or minutes of
    CPU. Keeping it would mean seeds that fail for reasons that have
    nothing to do with the interpreter. `\\dontshow{}` is ordinary code
    that is merely hidden from the reader, so its body stays.
    """
    for tag in ("dontrun", "donttest"):
        while True:
            m = re.search(r'\\' + tag + r'\s*\{', body)
            if not m:
                break
            body = body[:m.start()] + body[_matching_brace(body, m.end() - 1):]
    while True:
        m = re.search(r'\\dontshow\s*\{', body)
        if not m:
            break
        end = _matching_brace(body, m.end() - 1)
        body = body[:m.start()] + body[m.end():end - 1] + body[end:]
    # Rd escapes: \% is a literal percent (an unescaped one starts an Rd
    # comment), \\ is a backslash, \{ and \} are braces.
    body = body.replace("\\%", "%").replace("\\{", "{").replace("\\}", "}")
    body = re.sub(r'\\\\', r'\\', body)
    return body.strip("\n")


def _rd_example_seeds(src_root):
    out = []
    for pkg_man in sorted(_glob_dirs(os.path.join(src_root, "src", "library"), "man")):
        pkg = os.path.basename(os.path.dirname(pkg_man))
        for name in sorted(os.listdir(pkg_man)):
            if not name.endswith(".Rd"):
                continue
            path = os.path.join(pkg_man, name)
            try:
                text = open(path, encoding="utf-8", errors="replace").read()
            except OSError:
                continue
            m = re.search(r'\\examples\s*\{', text)
            if not m:
                continue
            body = text[m.end():_matching_brace(text, m.end() - 1) - 1]
            body = _strip_rd_markup(body)
            if not body.strip() or len(body) > _MAX_SEED_BYTES:
                continue
            stem = re.sub(r'[^A-Za-z0-9_.-]', '_', name[:-3])
            out.append((f"ex_{pkg}_{stem}.R", body + "\n"))
    return out


def _glob_dirs(root, leaf):
    found = []
    if not os.path.isdir(root):
        return found
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry, leaf)
        if os.path.isdir(p):
            found.append(p)
    return found


def _test_seeds(src_root):
    out = []
    roots = [("tests", os.path.join(src_root, "tests"))]
    roots += [(f"lib_{os.path.basename(os.path.dirname(p))}", p)
              for p in _glob_dirs(os.path.join(src_root, "src", "library"), "tests")]
    for label, root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(".R"):
                    continue
                path = os.path.join(dirpath, name)
                if os.path.getsize(path) > _MAX_SEED_BYTES:
                    continue
                text = open(path, encoding="utf-8", errors="replace").read()
                if not text.strip():
                    continue
                stem = re.sub(r'[^A-Za-z0-9_.-]', '_', name[:-2])
                out.append((f"{label}_{stem}.R", text))
    return out


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)
    total = 0
    for label, items in (("Rd examples", _rd_example_seeds(src_root)),
                         ("tests", _test_seeds(src_root))):
        for fname, content in items:
            with open(os.path.join(seeds, fname), "w", encoding="utf-8") as f:
                f.write(content)
        print(f"  {label}: {len(items)} seeds")
        total += len(items)
    print(f"Collected {total} seeds into {seeds}")
    return total


def _build_corpus(project_root):
    seeds = os.path.join(project_root, "seeds")
    if not os.path.isdir(seeds):
        return
    db = os.path.join(project_root, "corpus.db")
    if os.path.exists(db):
        os.remove(db)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ffl_r_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "r-src")
    install = os.path.join(project_root, "install")
    r_bin = os.path.join(install, "bin", "R")

    if not os.path.exists(r_bin):
        if not os.path.exists(os.path.join(src_root, "configure.ac")):
            print(f"Cloning R ({R_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {R_BRANCH} {R_REPO} {src_root}")
        cc = os.environ.get("CC", "clang-18")
        cxx = os.environ.get("CXX", "clang++-18")
        build = os.path.join(src_root, "build")
        env = dict(os.environ)
        env["ASAN_OPTIONS"] = "detect_leaks=0:use_sigaltstack=0"
        san = "" if os.environ.get("FFL_R_NO_ASAN") == "1" else \
            " -fsanitize=address -fno-omit-frame-pointer"
        # The `svnonly` make target insists on knowing the Subversion
        # revision, and the GitHub mirror is a git clone: `git svn info`
        # fails and the build stops with "ERROR: not an svn checkout".
        # The mirror does carry the revision — every commit message ends
        # in a `git-svn-id: ...@<rev>` line — so it is written into the
        # SVNINFO file the same rule falls back to.
        rev = "90000"
        try:
            body = subprocess.run(["git", "log", "-1", "--format=%b"], cwd=src_root,
                                  capture_output=True, text=True).stdout
            m = re.search(r'git-svn-id:\s*\S+@(\d+)', body)
            if m:
                rev = m.group(1)
        except Exception:
            pass
        with open(os.path.join(src_root, "SVNINFO"), "w") as f:
            f.write(f"Revision: {rev}\n"
                    "Last Changed Date: 2026-01-01 00:00:00 +0000 (Thu, 01 Jan 2026)\n")

        _run(f"""
set -e
cd {src_root}
[ -f configure ] || ./tools/rsync-recommended 2>/dev/null || true
[ -f configure ] || autoreconf -i
mkdir -p {build}
cd {build}
if [ ! -f Makefile ]; then
  ../configure --prefix={install} \\
    CC="{cc}{san}" CXX="{cxx}{san}" FC=gfortran F77=gfortran \\
    CFLAGS="-O1 -g" CXXFLAGS="-O1 -g" \\
    --enable-strict-barrier \\
    --with-recommended-packages=no \\
    --with-x=no --with-tcltk=no --with-cairo=no --with-aqua=no \\
    --disable-java --disable-nls --disable-openmp
fi
make -j{_jobs()}
make install
""", env=env)

    if not os.path.exists(r_bin):
        raise RuntimeError(f"R build failed: {r_bin} not found")
    print(subprocess.run([r_bin, "--version"], capture_output=True, text=True,
                         env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}
                         ).stdout.splitlines()[:1])

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"R setup complete. R: {r_bin}")


def setup_cov(project_root):
    setup(project_root)


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
