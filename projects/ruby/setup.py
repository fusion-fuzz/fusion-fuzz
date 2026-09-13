"""
projects/ruby/setup.py — build the Ruby interpreter under test and lift a
seed corpus out of its own test suite.

Called by main.py as setup(project_root). Leaves an instrumented
interpreter at projects/ruby/install/bin/ruby and seeds in
projects/ruby/seeds.

Build
-----
ruby/ruby trunk, configured per doc/contributing/building_ruby.md:

  * -DRUBY_DEBUG=1: the VM's own consistency checks (RUBY_ASSERT) are
    compiled in. Without it an internal contract violation is only seen
    when it later corrupts something visible.
  * AddressSanitizer (clang >= 18, -DUSE_MN_THREADS=0 as the doc requires).
  * -O1 rather than -O0: ASan startup cost dominates a short script's
    runtime and the corpus is ~5,000 small scripts.

YJIT/ZJIT need a Rust toolchain and are left out of the image; the
interpreter's two parsers (parse.y and prism) are both present, and the
driver alternates between them.

Seed corpus
-----------
Three sources, all from the cloned tree:

  1. bootstraptest/*.rb — `assert_equal %q{expected}, %q{program}` lines.
     The program argument is a complete top-level script; ~1,600 of them.
  2. test/ruby/test_*.rb — 6,800 test methods. Each is written out as a
     standalone script: the class's `setup` body, the helper methods and
     class-level definitions the test refers to, then the test body
     dedented to top level. The test/unit assertions it calls are
     provided by ffl_shim.rb at run time (see there). This is the Ruby
     analogue of the decomposed CPython corpus.
  3. sample/*.rb and benchmark/*.rb — whole programs.

Tests that need test/lib's process-spawning helpers, a C-extension test
helper, or concurrency are left out (config.yaml's exclude patterns catch
the ones that get through).
"""

import multiprocessing
import os
import re
import shutil
import subprocess
import textwrap

RUBY_REPO = "https://github.com/ruby/ruby.git"
RUBY_BRANCH = os.environ.get("FFL_RUBY_BRANCH", "master")

_MAX_SEED_BYTES = 24 * 1024
_MAX_TEST_LINES = 140


def _run(cmd, cwd=None, env=None):
    print(f"[run] {cmd[:220]}")
    subprocess.run(["bash", "-c", cmd], check=True, cwd=cwd, env=env)


def _jobs():
    env = os.environ.get("FFL_RUBY_JOBS")
    if env and env.isdigit() and int(env) > 0:
        return int(env)
    return max(1, min(multiprocessing.cpu_count() - 2, 12))


# ---------------------------------------------------------------------------
# bootstraptest
# ---------------------------------------------------------------------------

_BT_ASSERT_RE = re.compile(r'^(assert_(?:equal|match|not_match|normal_exit|finish))\b', re.M)
_BT_SKIP_FILES = ("test_yjit", "test_ractor", "test_thread", "test_fork",
                  "test_io", "test_env", "test_gc", "test_insns")


def _pct_q_args(text, start):
    """All `%q{...}` (or %q(...)/%q[...]) arguments of the statement that
    starts at `start`; returns (bodies, end_offset)."""
    bodies = []
    i = start
    n = len(text)
    while i < n:
        m = re.compile(r'%q([{(\[])').search(text, i)
        if not m:
            return bodies, n
        # A statement ends at a newline that is not inside an open %q.
        seg = text[i:m.start()]
        if bodies and '\n' in seg and not seg.strip().endswith(','):
            return bodies, i
        opener = m.group(1)
        closer = {"{": "}", "(": ")", "[": "]"}[opener]
        depth, j = 1, m.end()
        while j < n and depth:
            c = text[j]
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
            j += 1
        bodies.append(text[m.end():j - 1])
        i = j
    return bodies, n


def _bootstraptest_seeds(src_root):
    out = []
    bt = os.path.join(src_root, "bootstraptest")
    if not os.path.isdir(bt):
        return out
    for name in sorted(os.listdir(bt)):
        if not name.endswith(".rb") or name.startswith(_BT_SKIP_FILES):
            continue
        with open(os.path.join(bt, name), encoding="utf-8", errors="replace") as f:
            text = f.read()
        stem = name[:-3]
        for k, m in enumerate(_BT_ASSERT_RE.finditer(text)):
            bodies, _ = _pct_q_args(text, m.end())
            if not bodies:
                continue
            kind = m.group(1)
            # assert_equal/match/finish: (expected, program); normal_exit: (program)
            prog = bodies[0] if kind == "assert_normal_exit" or len(bodies) == 1 else bodies[1]
            prog = textwrap.dedent(prog.strip("\n"))
            if not prog.strip() or "#{" in prog:
                continue
            out.append((f"bt_{stem}_{k:04d}.rb", prog + "\n"))
    return out


# ---------------------------------------------------------------------------
# test/ruby decomposition
# ---------------------------------------------------------------------------

_SKIP_BODY_RE = re.compile(
    r'\bassert_separately\b|\bassert_in_out_err\b|\bassert_ruby_status\b|'
    r'\bassert_normal_exit\b|\bassert_no_memory_leak\b|\bEnvUtil\b|\bBug::|\bBug\.|'
    r'\bassert_ractor\b|\bRactor\b|\bThread\.|\bQueue\.new|\bMutex\.new|\bfork\b|'
    r'\bIO\.popen|\bProcess\.|\bSignal\.|\btrap\b|\bTempfile\b|\bDir\.mktmpdir|'
    r'\bFileUtils\b|\bSocket\b|\bTCP|\bUDP|\bsystem\(|`|\bexit!|\bsleep\b|\bgets\b|'
    r'\bSTDIN\b|\$stdin|\brequire\s+[\'"]-test-|\bRUBY_TEST|\b__FILE__\b|\b__dir__\b|'
    r'\bMKMF|\bmkmf\b|\bassert_linear_performance\b|\bassert_cpu_usage_low\b')

_DEF_RE = re.compile(r'^  def\s+(?:self\.)?([A-Za-z_]\w*[?!=]?)')
_CLASS_LEVEL_NAME_RE = re.compile(r'^  (?:class|module)\s+([A-Z]\w*)|^  ([A-Z]\w*)\s*=')
# `require_relative` can never resolve from the temp directory a child
# runs in, so only plain `require`s of stdlib names are kept.
_REQUIRE_RE = re.compile(r'^require\s+[\'"]([^\'"]+)[\'"]')
_DROP_REQUIRES = ("test/unit", "-test-", "envutil", "core_assertions", "minitest",
                  "rbconfig/sizeof", "tmpdir", "tempfile", "fileutils", "socket",
                  "open3", "timeout", "io/console", "io/wait", "pty", "etc",
                  "test/", "_test", "helper", "syntax_suggest", "coverage",
                  "objspace", "prism", "ripper", "rubygems", "bundler")


# A 2-space line that continues the member above it rather than starting
# a new one: the closing `end`/`]`/`}` of a multi-line member, a `rescue`
# or `ensure` clause of a def, a chained call.
_MEMBER_CONT_RE = re.compile(r'^(?:end\b|[\]})]|rescue\b|ensure\b|else\b|elsif\b|when\b|in\b|\.|&\.|&&|\|\|)')
_MEMBER_OPENS_RE = re.compile(r'^  (?:def|class|module)\b|(?:\bdo(?:\s*\|[^|]*\|)?|\{|\[|\()\s*$')


def _class_members(lines):
    """Split a test class body (2-space members) into (kind, name, text)
    blocks: 'def' members and everything else at class level."""
    members = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("  ") or line.startswith("   ") or not line.strip():
            i += 1
            continue
        # A member runs until the next line at exactly 2-space indent
        # that starts a new member. `end`, `]`, `}` and `rescue`/`ensure`
        # at 2 spaces belong to the member above them (a def with a
        # rescue clause, a multi-line constant), and one of `end`/`]`/`}`
        # closes a member that opened a block.
        opens = bool(_MEMBER_OPENS_RE.search(line))
        j = i + 1
        while j < n:
            l2 = lines[j]
            if l2.strip() and l2.startswith("  ") and not l2.startswith("   "):
                st = l2.strip()
                if _MEMBER_CONT_RE.match(st):
                    j += 1
                    if opens and re.match(r'^(?:end\b|[\]})])', st):
                        break
                    continue
                break
            j += 1
        block = lines[i:j]
        m = _DEF_RE.match(line)
        if m:
            members.append(("def", m.group(1), block))
        else:
            m2 = _CLASS_LEVEL_NAME_RE.match(line)
            name = (m2.group(1) or m2.group(2)) if m2 else None
            members.append(("stmt", name, block))
        i = j
    return members


def _dedent_body(block):
    """Lines strictly inside a `  def ... end` member, dedented by 4. A
    body with its own `rescue`/`ensure` clauses (at the def's indent) is
    kept as a `begin ... end` so the clauses still have a header."""
    inner = block[1:]
    if inner and inner[-1].strip() == "end":
        inner = inner[:-1]
    if any(re.match(r'^  (?:rescue|ensure)\b', l) for l in inner):
        return ["begin"] + [l[2:] if l.startswith("  ") else l for l in inner] + ["end"]
    out = []
    for l in inner:
        out.append(l[4:] if l.startswith("    ") else l.lstrip())
    return out


def _decompose_test_file(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    lines = text.splitlines()
    magic = [l for l in lines[:3] if re.match(r'^#\s*(?:frozen_string_literal|coding|encoding|-\*-)', l)]
    requires = []
    for l in lines:
        m = _REQUIRE_RE.match(l)
        if m and not any(d in m.group(1) for d in _DROP_REQUIRES):
            requires.append(l.strip())
    # The test class: from `class TestX < Test::Unit::TestCase` to the
    # column-0 `end`.
    seeds = []
    stem = os.path.basename(path)[:-3]
    i = 0
    while i < len(lines):
        if not re.match(r'^class\s+\w+(?:::\w+)*\s*<\s*Test::Unit::TestCase', lines[i]):
            i += 1
            continue
        j = i + 1
        while j < len(lines) and lines[j].rstrip() != "end":
            j += 1
        members = _class_members(lines[i + 1:j])
        i = j + 1
        setup = [m for m in members if m[0] == "def" and m[1] == "setup"]
        setup_body = _dedent_body(setup[0][2]) if setup else []
        helpers = {m[1]: m for m in members if m[0] == "def"
                   and not m[1].startswith("test_") and m[1] not in ("setup", "teardown")}
        stmts = [m for m in members if m[0] == "stmt" and m[1]]
        for kind, name, block in members:
            if kind != "def" or not name.startswith("test_"):
                continue
            body = _dedent_body(block)
            if not body or len(body) > _MAX_TEST_LINES:
                continue
            body_text = "\n".join(body)
            if _SKIP_BODY_RE.search(body_text) or _SKIP_BODY_RE.search("\n".join(setup_body)):
                continue
            # Helper methods and class-level definitions the body names.
            used = set(re.findall(r'[A-Za-z_]\w*[?!]?', body_text))
            pre = []
            for hname, (_, _, hblock) in helpers.items():
                if hname in used or hname.rstrip("?!=") in used:
                    if len(hblock) <= 40 and not _SKIP_BODY_RE.search("\n".join(hblock)):
                        pre.extend(l[2:] if l.startswith("  ") else l for l in hblock)
                        pre.append("")
            for _, sname, sblock in stmts:
                if sname in used and len(sblock) <= 60:
                    pre.extend(l[2:] if l.startswith("  ") else l for l in sblock)
                    pre.append("")
            parts = magic + requires + ([""] if (magic or requires) else [])
            parts += pre + setup_body + ([""] if setup_body else []) + body
            content = "\n".join(parts).rstrip() + "\n"
            if len(content) > _MAX_SEED_BYTES:
                continue
            seeds.append((f"{stem}__{name}.rb", content))
    return seeds


def _test_ruby_seeds(src_root):
    out = []
    root = os.path.join(src_root, "test", "ruby")
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        if not (name.startswith("test_") and name.endswith(".rb")):
            continue
        if name in ("test_thread.rb", "test_fiber.rb", "test_process.rb", "test_signal.rb",
                    "test_io.rb", "test_file.rb", "test_file_exhaustive.rb", "test_dir.rb",
                    "test_dir_m17n.rb", "test_gc.rb", "test_gc_compact.rb", "test_env.rb",
                    "test_rubyoptions.rb", "test_argf.rb", "test_readpartial.rb",
                    "test_io_buffer.rb", "test_io_timeout.rb", "test_thread_queue.rb",
                    "test_thread_cv.rb", "test_beginendblock.rb", "test_syntax_suggest.rb",
                    "test_autoload.rb", "test_require.rb", "test_require_lib.rb",
                    "test_system.rb", "test_rubyvm_jit.rb", "test_yjit.rb", "test_zjit.rb",
                    "test_yjit_exit_locations.rb", "test_pty.rb", "test_pipe.rb",
                    "test_fiber_scheduler.rb", "test_ractor.rb", "test_notimp.rb"):
            continue
        try:
            out.extend(_decompose_test_file(os.path.join(root, name)))
        except Exception as e:  # a parse oddity in one file must not kill the corpus
            print(f"warning: {name}: {e}")
    return out


# ---------------------------------------------------------------------------
# sample/ and benchmark/
# ---------------------------------------------------------------------------

def _whole_program_seeds(src_root):
    out = []
    for sub in ("sample", "benchmark"):
        root = os.path.join(src_root, sub)
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not name.endswith(".rb") or not os.path.isfile(p):
                continue
            if os.path.getsize(p) > _MAX_SEED_BYTES:
                continue
            with open(p, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if "__END__" in text or text.startswith("#!") and "-x" in text[:80]:
                continue
            out.append((f"{sub}_{name}", text))
    return out


def _collect_seeds(project_root, src_root):
    seeds = os.path.join(project_root, "seeds")
    shutil.rmtree(seeds, ignore_errors=True)
    os.makedirs(seeds, exist_ok=True)
    groups = [("bootstraptest", _bootstraptest_seeds(src_root)),
              ("test/ruby", _test_ruby_seeds(src_root)),
              ("sample+benchmark", _whole_program_seeds(src_root))]
    total = 0
    for label, items in groups:
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
        "ffl_ruby_parser_setup", os.path.join(project_root, "parser.py"))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.collect_seeds(seeds)


# ---------------------------------------------------------------------------

def setup(project_root):
    project_root = os.path.abspath(project_root)
    src_root = os.path.join(project_root, "ruby-src")
    install = os.path.join(project_root, "install")
    ruby_bin = os.path.join(install, "bin", "ruby")

    if not os.path.exists(ruby_bin):
        if not os.path.exists(os.path.join(src_root, "configure.ac")):
            print(f"Cloning Ruby ({RUBY_BRANCH}) ...")
            shutil.rmtree(src_root, ignore_errors=True)
            _run(f"git clone --depth=1 --branch {RUBY_BRANCH} {RUBY_REPO} {src_root}")
        cc = os.environ.get("CC", "clang-18")
        build = os.path.join(src_root, "build")
        env = dict(os.environ)
        # miniruby runs during the build; under ASan it must not report
        # leaks or the build stops at the first one.
        env["ASAN_OPTIONS"] = "detect_leaks=0:use_sigaltstack=0"
        _run(f"""
set -e
cd {src_root}
[ -f configure ] || ./autogen.sh
mkdir -p {build}
cd {build}
if [ ! -f Makefile ]; then
  ../configure --prefix={install} --disable-install-doc \\
    CC={cc} \\
    cflags="-fsanitize=address -fno-omit-frame-pointer -DUSE_MN_THREADS=0" \\
    cppflags="-DRUBY_DEBUG=1" optflags="-O1" debugflags="-g"
fi
make -j{_jobs()}
make install
""", env=env)

    if not os.path.exists(ruby_bin):
        raise RuntimeError(f"Ruby build failed: {ruby_bin} not found")
    print(subprocess.run([ruby_bin, "--disable-gems", "-v"], capture_output=True, text=True,
                         env={**os.environ, "ASAN_OPTIONS": "detect_leaks=0"}).stdout.strip())

    _collect_seeds(project_root, src_root)
    _build_corpus(project_root)
    print(f"Ruby setup complete. ruby: {ruby_bin}")


def setup_cov(project_root):
    """Coverage build (--setup-cov): reuses the normal build."""
    setup(project_root)


if __name__ == "__main__":
    setup(os.path.dirname(os.path.abspath(__file__)))
