"""
projects/julia/setup.py — build the Julia compiler/runtime under test and
lift a seed corpus out of its own tests and documentation.

Called by main.py as setup(project_root). Leaves an interpreter at
projects/julia/julia-src/julia and seeds in projects/julia/seeds.

Build
-----
JuliaLang/julia master with:

  * `FORCE_ASSERTIONS=1` — Julia's own C/C++ assertions. This is the
    whole point of building rather than downloading: a release Julia
    carries on past a broken invariant, an assert build says where it
    broke. It costs nothing at build time.
  * prebuilt dependencies (the default `USE_BINARYBUILDER=1`): LLVM,
    OpenBLAS, SuiteSparse and the rest are downloaded rather than
    built, which is the difference between a 40-minute build and a
    day-long one. `LLVM_ASSERTIONS=1` would force LLVM to be built from
    source and is deliberately not set.
  * `JULIA_PRECOMPILE=0` — the system image is still built (Julia cannot
    run without one), but the extra package precompilation that follows
    it is skipped.

Seed corpus
-----------
Two sources, both from the cloned tree:

  1. `@testset "name" begin ... end` blocks in test/*.jl and
     stdlib/*/test/*.jl — 2,400 of them. Each is written out as a
     standalone script with `using Test` in front: the block is exactly
     one self-contained unit of behaviour, which is what makes this the
     Julia analogue of the decomposed CPython corpus.
  2. ```jldoctest``` blocks in the docstrings under base/ and stdlib/ —
     790 more. The `julia> ` prompts become statements and the expected
     output lines are dropped, so each becomes a short script written by
     the people who wrote the function.
"""

import multiprocessing
import os
import re
import shutil
import subprocess

JULIA_REPO = "https://github.com/JuliaLang/julia"
JULIA_BRANCH = os.environ.get("FFL_JULIA_BRANCH", "master")

_MAX_SEED_BYTES = 24 * 1024
# Tests that need a network, a display, many cores, or minutes of CPU.
_SKIP_FILES = ("threads", "distributed", "channels", "download", "sockets",
               "spawn", "file", "path", "cmdlineargs", "loading", "precompile",
               "misc", "backtrace", "stress", "gcext", "clangsa", "llvmpasses",
               "relocatedepot", "docs", "Distributed", "Sockets", "Downloads",
               "REPL", "InteractiveUtils", "Profile", "Pkg")
_SKIP_BODY_RE = re.compile(
    r'@test_broken|Threads\.|@spawn|@async|@distributed|addprocs|remotecall|'
    r'run\(`|read\(`|open\(`|download\(|Sockets\.|listen\(|connect\(|'
    r'mktempdir|mktemp|tempname|rm\(|touch\(|write\(|Base\.Filesystem|'
    r'ccall\(|unsafe_(?:load|store|wrap|pointer)|pointer_from_objref|'
    r'@allocated|@time|sleep\(|exit\(|error\("test|Sys\.which|Libdl\.')


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_JULIA_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


# ---------------------------------------------------------------------------
# @testset blocks
# ---------------------------------------------------------------------------

_TESTSET_RE = re.compile(r'^@testset\s+(?:"([^"\n]*)"|[^\n]*?)\s*begin\s*$', re.M)


def _block_end(lines, start):
    """Index just past the `end` that closes the block opening at `start`.

    Counts Julia's block keywords rather than braces. An opener only
    counts at the start of a statement (or after `=`, `(`, `,`), which
    keeps the modifier forms (`x = y for i in z`) and the `end` used as
    an index (`a[end]`) out of the count.
    """
    depth = 0
    opener = re.compile(r'(?:^|[=({,]|\bdo\b)\s*'
                        r'(?:function|begin|if|for|while|let|try|struct|module|'
                        r'macro|quote|mutable\s+struct)\b')
    end_re = re.compile(r'(?<![\w.\[])end\b(?!\s*\])')
    for i in range(start, len(lines)):
        line = lines[i].split("#", 1)[0]
        depth += len(opener.findall(line))
        if i == start:
            depth = max(depth, 1)
        depth -= len(end_re.findall(line))
        if depth <= 0:
            return i + 1
    return len(lines)


# Assertion macros, rewritten to the bare expression. A fused program
# deliberately changes what a value is, so the *original* expected value
# is no longer meaningful — and `@test` turning a changed value into an
# uncaught `Test.FallbackTestSetException` made every real dataflow edge
# count as invalid (11.4% valid on the first measurement, against 97%
# for the two strategies that do not change values). The expression
# still runs; only the comparison against a stale expectation goes.
# This is the same step the Ruby adapter's ffl_shim.rb performs at run
# time; here it is done in the corpus because Julia's `using Test`
# would shadow anything defined beforehand.
# The value digest. Julia's test bodies print nothing, so every valid
# child of a fused pair produced byte-identical (empty) output and the
# diversity proxy saw 3 distinct outputs among 300 children — the same
# blindness the Ruby shim's assertion digest and the TypeScript emit
# digest fixed for those targets. Each seed therefore folds the value of
# every assertion it used to make into a running hash and prints it at
# exit, so two children that computed different values are visibly
# different programs.
#
# Three details that are not free choices. The value is hashed through
# `sprint(show, x)` rather than `hash(x)`: `hash` falls back to
# `objectid` for a type that does not define one, and an object id
# varies between runs, which would manufacture diversity rather than
# measure it. The digest is spelled in the letters g..z because the
# validity harness masks digits and long hex runs out of a fingerprint
# before comparing it. And it is one physical line, so a state cut in a
# chain cannot land in the middle of it.
# Every call in it is written `Base.f`, and its parameter is `fflx`:
# dataflow fusion renames call sites, it never renames a name that
# follows a dot, and the three bare names it could still reach (`FFLD`,
# `fflv`, `fflx`) are in JuliaFusionStrategy._DATAFLOW_KEYWORDS. Without
# that the instrumentation is itself a rename target, and a child would
# fail on the probe rather than on the program.
_DIGEST_LINE = (
    'const FFLD = Base.Ref(Base.UInt64(0)); '
    'fflv(fflx) = (try; FFLD[] = Base.hash(Base.sprint(fflo -> Base.show('
    'Base.IOContext(fflo, :limit => true, :displaysize => (6, 80)), fflx)), FFLD[]); '
    'catch; FFLD[] = Base.hash(Base.string(Base.typeof(fflx)), FFLD[]); end; fflx); '
    'fflw() = (for fflk in Base.names(Main; all=true); ffls = Base.string(fflk); '
    '(Base.startswith(ffls, "#") || Base.startswith(ffls, "ffl") || '
    'ffls in ("Main", "Base", "Core", "ans", "eval", "include", "FFLD")) && continue; '
    'try; fflu = Base.getglobal(Main, fflk); '
    '(fflu isa Base.Module || fflu isa Base.Function || fflu isa Base.DataType) || fflv(fflu); '
    'catch; end; end); '
    'Base.atexit(() -> try; fflw(); Base.println("ffl-", Base.String('
    '[Base.Char(103 + fflk) for fflk in Base.digits(FFLD[], base=20, pad=13)])); catch; end)'
)


_ASSERT_LINE_RE = re.compile(
    r'^(?P<indent>[ \t]*)@(?P<macro>test_throws|test_broken|test_skip|test_logs|'
    r'test_deprecated|test_warn|test_nowarn|inferred|test)\b(?P<rest>.*)$')


def _split_trailing_comment(text):
    """(code, comment) for one line, splitting at a `#` that is not
    inside a string or a character literal. `@test f(x) == y  # why`
    became `fflv(f(x) == y  # why)` when the comment was left in place,
    and the closing parenthesis vanished into it."""
    i, n = 0, len(text)
    quote = None
    while i < n:
        c = text[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if text.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        elif text.startswith('"""', i):
            quote = '"""'
            i += 3
            continue
        elif c in "\"'":
            quote = c
            i += 1
            continue
        elif c == "#":
            return text[:i].rstrip(), text[i:]
        i += 1
    return text.rstrip(), ""


# An expression that ends with a binary operator, a comma or `::`
# continues on the next line. Wrapping such a line closes the
# parenthesis in the middle of the expression:
#   @test hash(x, SEED) ==
#         hash(y, SEED)
# became `fflv(hash(x, SEED) ==)`. Those lines keep their `@test`.
_CONTINUES_RE = re.compile(
    r'(?:[-+*/\\^%<>&|~,.=:?]|&&|\|\||->|==|!=|<=|>=|::|\bin|\bfor|\bif)\s*$')


def _balanced(text):
    return (text.count("(") == text.count(")") and text.count("[") == text.count("]")
            and text.count("{") == text.count("}"))


def _rewrite_assertions(body):
    """`@test expr` → `expr`, and the throwing forms → a `try` block."""
    out = []
    for line in body.splitlines():
        m = _ASSERT_LINE_RE.match(line)
        if not m or not _balanced(m.group("rest")):
            out.append(line)
            continue
        rest_raw, trailing = _split_trailing_comment(m.group("rest"))
        trailing = ("  " + trailing) if trailing else ""
        indent, macro, rest = m.group("indent"), m.group("macro"), rest_raw.strip()
        if _CONTINUES_RE.search(rest):
            out.append(line)          # a multi-line expression; leave it whole
            continue
        if macro in ("test_broken", "test_skip"):
            continue                                  # marked as not expected to work
        if macro == "test_throws":
            # `@test_throws SomeError expr` — the type comes first. The
            # exception is part of what the program computed, so its
            # type goes into the digest too.
            parts = rest.split(None, 1)
            expr = parts[1] if len(parts) > 1 else ""
            out.append(f"{indent}try; fflv({expr}); catch e; fflv(typeof(e)); end{trailing}"
                       if expr.strip() else line)
            continue
        if macro in ("test_logs", "test_deprecated", "test_warn", "test_nowarn"):
            continue                                  # asserts on emitted text
        if not rest:
            continue
        out.append(f"{indent}fflv({rest}){trailing}")
    return "\n".join(out)


# The `using` lines the file was run with. A `@testset` body lifted out
# of `stdlib/LinearAlgebra/test/cholesky.jl` calls `cholesky`, `randn`
# and `RowMaximum`, none of which a bare script has: 1,538 of the 1,966
# seeds that failed on their own came from a file with a `using` line
# the seed did not carry. Module-relative imports (`using .Foo`,
# `using Main.LinearAlgebraTestHelpers.SizedArrays`) are dropped — they
# name a module the test runner built, which a standalone script has no
# way to reach.
_USING_RE = re.compile(r'^[ \t]*(using|import)\s+(?P<what>[^\n]*?)\s*$')
_USING_NAME_RE = re.compile(r'^[A-Za-z_][\w]*(\.[A-Za-z_]\w*)*$')
# Only these may be carried. It is the same list config.yaml's
# `using` exclusion allows, and for the same reason: anything else
# names a package this build does not ship, and carrying it would make
# the seed excluded from the corpus rather than runnable — which is how
# the first attempt lost most of the corpus without saying so.
_USING_ALLOWED = frozenset((
    "Test", "Base", "LinearAlgebra", "Random", "Printf", "Dates",
    "Statistics", "Unicode", "Markdown", "Serialization", "SparseArrays",
    "Logging",
))


def _file_usings(lines):
    """The file's own `using`/`import` lines, in order, with the
    module-relative ones removed. A line ending in a comma continues on
    the next one (`using LinearAlgebra: BlasComplex,` ...)."""
    out, seen = [], set()
    i = 0
    while i < len(lines):
        m = _USING_RE.match(lines[i])
        if not m:
            i += 1
            continue
        what, j = m.group("what"), i
        while what.rstrip().endswith(",") and j + 1 < len(lines):
            j += 1
            what = what.rstrip() + " " + lines[j].strip()
        i = j + 1
        if what.startswith("."):
            continue
        head, _, members = what.partition(":")
        names = [n.strip() for n in head.split(",")]
        keep = [n for n in names
                if _USING_NAME_RE.match(n) and not n.startswith(("Main.", "."))
                and n != "Test" and n.split(".")[0] in _USING_ALLOWED]
        if not keep or len(keep) != len(names):
            # A mixed line (`using Test, Main.Helpers`) is rebuilt from
            # the names that survive; an empty one is dropped.
            if not keep:
                continue
        text = f"{m.group(1)} {', '.join(keep)}"
        if members.strip() and len(keep) == 1:
            text += f": {members.strip()}"
        if text not in seen:
            seen.add(text)
            out.append(text)
    return out


_SETUP_DEF_RE = re.compile(
    r'^(?:const\s+|global\s+)?([A-Za-z_][\w!]*)\s*(?:::[^=\n]+)?=(?![=>])|'
    r'^(?:mutable\s+)?struct\s+([A-Za-z_]\w*)|^abstract\s+type\s+([A-Za-z_]\w*)|'
    r'^function\s+([A-Za-z_][\w!]*)|^([A-Za-z_][\w!]*)\s*\([^)\n]*\)\s*=(?![=>])')


def _file_setup(lines):
    """[(names defined, text)] for the file's own top-level definitions,
    outside every `@testset`.

    A block lifted out of a test file still refers to what the file set
    up around it, so those statements travel with it. Multi-line
    definitions (a `struct`, a `function`) are kept whole.
    """
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TESTSET_RE.match(line):
            i = _block_end(lines, i)
            continue
        m = _SETUP_DEF_RE.match(line)
        if not m:
            i += 1
            continue
        names = {g for g in m.groups() if g}
        if re.match(r'^(?:mutable\s+)?struct\b|^abstract\s+type\b|^function\b', line) \
                and not re.search(r'\bend\s*$', line):
            end = _block_end(lines, i)
        else:
            end = i + 1
        block = "\n".join(lines[i:end])
        if len(block) <= 2000:
            out.append((names, block, i))
        i = end
    return out


def _testset_seeds(src_root):
    out = []
    roots = [os.path.join(src_root, "test")]
    stdlib = os.path.join(src_root, "stdlib")
    if os.path.isdir(stdlib):
        for entry in sorted(os.listdir(stdlib)):
            t = os.path.join(stdlib, entry, "test")
            if os.path.isdir(t):
                roots.append(t)
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(".jl") or name.startswith(_SKIP_FILES):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                lines = text.splitlines()
                stem = re.sub(r'[^A-Za-z0-9_]', '_', name[:-3])
                setup = _file_setup(lines)
                usings = _file_usings(lines)
                for k, m in enumerate(_TESTSET_RE.finditer(text)):
                    start = text[:m.start()].count("\n")
                    end = _block_end(lines, start)
                    body = "\n".join(lines[start:end])
                    if len(body) > _MAX_SEED_BYTES or _SKIP_BODY_RE.search(body):
                        continue
                    if body.count("\n") < 2:
                        continue
                    # The block's *body*, at top level, not the block.
                    # `@testset ... begin ... end` is a scope of its own,
                    # so leaving it wrapped would mean every name the
                    # seed defines is invisible to the other half of a
                    # fused program — 74% of pairs got no dataflow edge
                    # at all when the wrapper was kept. `@test` works
                    # perfectly well at top level.
                    inner = lines[start + 1:end - 1]
                    inner = [l[4:] if l.startswith("    ") else l for l in inner]
                    body = "\n".join(inner).strip("\n")
                    if not body.strip() or body.count("\n") < 1:
                        continue
                    # Whatever the file defined above the block and the
                    # block refers to. A `@testset` body is written
                    # against the file around it — the first one taken
                    # this way failed on `UndefVarError: A not defined`,
                    # which says nothing about Julia. This is the same
                    # step the CPython and Ruby corpora needed.
                    used = set(re.findall(r'[A-Za-z_][\w!]*', body))
                    # Only what was defined *before* this block, and for
                    # a name defined more than once only the definition
                    # that was in force here — a test file rebinds `A`
                    # half a dozen times, and carrying them all would
                    # hand the block the wrong one.
                    chosen = {}
                    for names, text_, at in setup:
                        if at >= start or not (names & used):
                            continue
                        for n in names & used:
                            chosen[n] = (at, text_)
                    pre = [t for _, t in sorted(dict(
                        (at, t) for at, t in chosen.values()).items())]
                    body = _rewrite_assertions(body)
                    if not body.strip():
                        continue
                    parts = ["using Test"] + usings + [_DIGEST_LINE, ""] + pre + \
                        ([""] if pre else []) + [body]
                    content = "\n".join(parts).strip("\n") + "\n"
                    if len(content) > _MAX_SEED_BYTES:
                        continue
                    out.append((f"ts_{stem}_{k:04d}.jl", content))
    return out


# ---------------------------------------------------------------------------
# jldoctest blocks
# ---------------------------------------------------------------------------

_DOCTEST_RE = re.compile(r'```jldoctest[^\n]*\n(.*?)```', re.S)


def _doctest_script(block):
    """The statements of a jldoctest, without prompts or expected output."""
    out, cont = [], False
    for line in block.splitlines():
        s = line.rstrip()
        if s.startswith("julia> "):
            out.append(s[7:])
            cont = True
        elif s.startswith("       ") and cont:
            out.append(s[7:])              # continuation of the last statement
        else:
            cont = False                   # an expected-output line
    return "\n".join(out).strip()


def _doctest_seeds(src_root):
    out = []
    for sub in ("base", "stdlib"):
        root = os.path.join(src_root, sub)
        if not os.path.isdir(root):
            continue
        for dirpath, _dirs, files in os.walk(root):
            for name in sorted(files):
                if not name.endswith(".jl"):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    text = open(path, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                if "jldoctest" not in text:
                    continue
                stem = re.sub(r'[^A-Za-z0-9_]', '_', name[:-3])
                for k, m in enumerate(_DOCTEST_RE.finditer(text)):
                    script = _doctest_script(m.group(1))
                    if not script or len(script) > _MAX_SEED_BYTES:
                        continue
                    if _SKIP_BODY_RE.search(script) or script.count("\n") < 1:
                        continue
                    # The digest goes in here too: a jldoctest has no
                    # assertions to fold, and a `julia>` statement in a
                    # script does not echo, so without it every valid
                    # child of a doctest pair looks identical.
                    out.append((f"dt_{sub}_{stem}_{k:04d}.jl",
                                _DIGEST_LINE + "\n\n" + script + "\n"))
    return out


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)
    total = 0
    for label, items in (("@testset blocks", _testset_seeds(src_root)),
                         ("jldoctest blocks", _doctest_seeds(src_root))):
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
        "ffl_julia_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "julia-src")
    julia_bin = os.path.join(src_root, "julia")

    if not os.path.exists(julia_bin):
        if not os.path.exists(os.path.join(src_root, "Makefile")):
            print(f"Cloning Julia ({JULIA_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {JULIA_BRANCH} {JULIA_REPO} {src_root}")
        with open(os.path.join(src_root, "Make.user"), "w") as f:
            f.write("FORCE_ASSERTIONS=1\n"
                    "JULIA_PRECOMPILE=0\n"
                    "USE_BINARYBUILDER=1\n")
        _run(f"cd {src_root} && make -j{_jobs()}")

    if not os.path.exists(julia_bin):
        raise RuntimeError(f"Julia build failed: {julia_bin} not found")
    print(subprocess.run([julia_bin, "--version"], capture_output=True,
                         text=True).stdout.strip())

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"Julia setup complete. julia: {julia_bin}")


def setup_cov(project_root):
    setup(project_root)


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
