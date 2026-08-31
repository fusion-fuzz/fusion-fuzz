"""
Tests for the CPython adapter's oracle.

CPython is the only target in this repo whose seeds are *executed* rather
than compiled, which inverts the usual question. For a compiler adapter
the hard part is telling a crash from a diagnostic; here almost every
failure is an ordinary Python exception that says the fusion produced
nonsense, and the hard part is finding the few exceptions that are
genuinely interpreter bugs.
"""

import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("ffl_cpython_analyzer_test", "projects/cpython/analyzer.py")


# ---------------------------------------------------------------------------
# The two assertion spellings
# ---------------------------------------------------------------------------

GLIBC_ASSERT = ("python: Objects/dictobject.c:1234: insertdict: "
                "Assertion `value != NULL' failed.\n")
CPYTHON_ASSERT = ('Objects/object.c:275: _Py_NegativeRefcount: '
                  'Assertion "op->ob_refcnt > 0" failed: object has negative ref count\n')


@pytest.mark.parametrize("output,where", [
    (GLIBC_ASSERT, "Objects/dictobject.c:1234"),
    (CPYTHON_ASSERT, "Objects/object.c:275"),
])
def test_both_assertion_spellings_are_recognised(output, where):
    """glibc closes with a single quote (`expr'), CPython's own
    _PyObject_AssertFailed uses double quotes ("expr").

    The previous driver matched ``Assertion `expr` failed`` — a backtick on
    *both* sides — which is neither of them, so every assertion failure in
    a --with-pydebug build fell through to the generic signature.
    """
    v = analyzer.classify(output)
    assert v["is_bug"] and v["kind"] == "assert"
    assert where in v["signature"]


# ---------------------------------------------------------------------------
# The bug class that hides inside a normal traceback
# ---------------------------------------------------------------------------

C_API_VIOLATION = (
    'Traceback (most recent call last):\n'
    '  File "t.py", line 3, in <module>\n'
    'SystemError: <built-in function foo> returned NULL without setting an exception\n')

ORDINARY_EXCEPTION = (
    'Traceback (most recent call last):\n'
    '  File "t.py", line 1, in <module>\n'
    'TypeError: unsupported operand type(s)\n')


def test_c_api_contract_violation_is_a_bug():
    """A C function returning NULL without setting an exception is a
    C-level contract violation — a real CPython (or extension) bug that
    arrives looking exactly like an ordinary traceback. On a fuzzer that
    executes its input this is the most reachable interpreter bug there is,
    and the previous oracle, which only looked for crashes, missed the
    whole class."""
    v = analyzer.classify(C_API_VIOLATION)
    assert v["is_bug"] and v["kind"] == "c-api"
    assert "returned NULL without setting an exception" in v["signature"]


def test_ordinary_exceptions_are_not_bugs():
    """Fusing two unrelated programs produces nonsense; a TypeError says so
    and says nothing about CPython."""
    assert analyzer.classify(ORDINARY_EXCEPTION)["is_bug"] is False
    assert analyzer.classify(ORDINARY_EXCEPTION)["kind"] == "exception"


@pytest.mark.parametrize("output", [
    "Traceback (most recent call last):\nMemoryError\n",
    "RecursionError: maximum recursion depth exceeded\n",
])
def test_resource_exceptions_are_not_bugs(output):
    """MemoryError and RecursionError are catchable exceptions, so they
    cannot be excluded by the crash-pattern matching alone — a fused
    program nests and allocates arbitrarily, and both are facts about the
    input."""
    v = analyzer.classify(output)
    assert v["is_bug"] is False and v["kind"] == "resource"


# ---------------------------------------------------------------------------
# The crash classes the old config never looked for
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("output,kind", [
    ("Segmentation fault (core dumped)\n", "signal"),
    ("Fatal Python error: Segmentation fault\n", "fatal"),
    ("Debug memory block at address p=0x5555: API 'o'\n    bad trailing pad byte\n", "memory"),
    ("SUMMARY: AddressSanitizer: heap-use-after-free Objects/listobject.c:123 in x\n", "sanitizer"),
    ("Objects/obmalloc.c:2431: runtime error: load of misaligned address\n", "ubsan"),
])
def test_real_crashes_are_reported(output, kind):
    """A plain segfault was not reported at all before: core/driver.py's
    _check_crash is purely pattern-based (its return-code check is
    commented out) and the old crash_patterns held only "SUMMARY:" and
    ": Assertion `", so the driver's own handler for "Segmentation fault"
    was dead code."""
    v = analyzer.classify(output)
    assert v["is_bug"] is True and v["kind"] == kind
    assert v["signature"]


def test_sanitizer_report_wins_over_the_abort_that_follows_it():
    combined = ("SUMMARY: AddressSanitizer: heap-use-after-free x.c:1 in f\n"
                "Aborted (core dumped)\n")
    assert analyzer.classify(combined)["kind"] == "sanitizer"


def test_signature_is_stable_across_seeds():
    a = GLIBC_ASSERT
    b = GLIBC_ASSERT.replace("python:", "/tmp/ffl/abc123/python:")
    assert analyzer.crash_signature(a) == analyzer.crash_signature(b)
    assert analyzer.crash_signature(ORDINARY_EXCEPTION) is None


# ---------------------------------------------------------------------------
# Containment — the concern unique to a target whose seeds execute
# ---------------------------------------------------------------------------

def test_seed_analysis_flags_what_must_not_run_concurrently():
    facts = analyzer.analyze_seed(
        "import socket\ns = socket.socket()\ns.bind(('', 8080))\n")
    assert facts["uses_network"]
    facts = analyzer.analyze_seed("import subprocess\nsubprocess.run(['ls'])\n")
    assert facts["uses_subprocess"]
    facts = analyzer.analyze_seed("import ctypes\n")
    assert facts["uses_ctypes"]
    assert not analyzer.analyze_seed("x = 1 + 2\n")["uses_network"]


def test_config_covers_what_the_analyzer_can_find():
    """crash_patterns is the coarse net core/driver.py applies before the
    analyzer runs; anything the analyzer can classify has to survive it."""
    import yaml
    with open(os.path.join(ROOT, "projects", "cpython", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    patterns = cfg["analysis"]["crash_patterns"]
    for probe in ("Segmentation fault (core dumped)",
                  "Fatal Python error: x",
                  "Assertion `x' failed",
                  "Debug memory block at address p=0x1",
                  "SystemError: f returned NULL without setting an exception"):
        assert any(p in probe for p in patterns), probe
    # And the executed-target notion of "expected failure" exists at all.
    assert cfg["analysis"]["syntax_patterns"]
    assert cfg["execution"]["concurrency"] > 1


def test_parser_uses_the_shared_base_and_emits_the_needed_keys():
    parser = _load("ffl_cpython_parser_test", "projects/cpython/parser.py")
    from core.parser import BaseParser
    assert issubclass(parser.CPythonParser, BaseParser), \
        "the parser reimplemented BaseParser's collection and sqlite schema"
    meta = parser._parser.parse_content(
        "import ctypes\nx = 1\ny = x + 2\nprint(y)\n", "test_foo.py")
    assert meta["type"] == "python"          # keys the dryrun collector uses
    assert meta["variables"] and meta["dataflows"]
    assert meta["uses_ctypes"] is True
    assert meta["is_test"] is True


def test_dataflow_analysis_uses_a_real_ast():
    """The C-family parser guesses dataflow from line co-occurrence; this
    one walks an AST, which is better information and worth keeping."""
    parser = _load("ffl_cpython_parser_ast_test", "projects/cpython/parser.py")
    variables, dataflows = parser.PythonFastDataflow().analyze(
        "a = 1\nb = a + 1\nunrelated = 2\n")
    assert "a" in variables and "b" in variables
    assert any({"a", "b"} <= set(g) for g in dataflows)
    # A syntax error yields nothing rather than raising.
    assert parser.PythonFastDataflow().analyze("def (:\n") == ([], [])


# ---------------------------------------------------------------------------
# Assembly: the plumbing that decides whether a fused child parses at all
# ---------------------------------------------------------------------------

import random  # noqa: E402

from core.fusion import Seed, get_strategies  # noqa: E402


def _cpython_strategy(**flags):
    return get_strategies("cpython", pre_analysis_enabled=True, **flags)[0]


def test_nested_imports_are_left_where_they_are():
    """Hoisting an indented import empties the block it came from.

        try:
            import _md5
        except ImportError:
            ...

    becomes `try:` immediately followed by `except`, i.e.
    "IndentationError: expected an indented block after 'try' statement" —
    which was the single largest cause of invalid children. It also turns a
    conditional import into a hard dependency.
    """
    s = _cpython_strategy(dataflow_fusion=True)
    src = "import os\ntry:\n    import _md5\nexcept ImportError:\n    _md5 = None\n"
    body, imports = s._extract_imports_and_body(src)
    assert imports == ["import os"]
    assert "import _md5" in body, body
    import ast as _ast
    _ast.parse(body)          # the try block still has a body


def test_parenthesised_imports_are_hoisted_whole():
    """`from x import (` spans several lines. Matching only the first
    hoists the opening and leaves the closing behind, so both halves are
    syntax errors — 53 of ~70 remaining parse failures before this."""
    s = _cpython_strategy(dataflow_fusion=True)
    src = 'from typing import (\n    IO,\n    Any,\n)\n\nx = 1\n'
    body, imports = s._extract_imports_and_body(src)
    assert len(imports) == 1 and imports[0].count("\n") == 3, imports
    assert "typing" not in body and "IO" not in body
    import ast as _ast
    _ast.parse(body)
    _ast.parse("\n".join(imports))


def test_decorators_stay_attached_to_what_they_decorate():
    """A column-zero decorator followed by a column-zero `def` is one
    statement, not two. Splitting them leaves a decorator with nothing to
    decorate."""
    s = _cpython_strategy(dataflow_fusion=True)
    blocks = s._py_toplevel_blocks("@property\ndef f(self):\n    return 1\n\nx = 2\n")
    assert len(blocks) == 2, blocks
    assert blocks[0].startswith("@property") and "def f" in blocks[0]


def test_try_except_is_one_block():
    """`except` at column zero continues the `try` above it."""
    s = _cpython_strategy(dataflow_fusion=True)
    blocks = s._py_toplevel_blocks("try:\n    x = 1\nexcept ValueError:\n    x = 2\n")
    assert len(blocks) == 1, blocks
    import ast as _ast
    _ast.parse(blocks[0])


def test_indent_fallback_handles_what_the_ast_path_would():
    """The host interpreter's `ast` is older than the CPython under test —
    3.12 here against a 3.16 trunk — so seeds using newer syntax take the
    fallback. It runs on real, valid code and has to keep decorators and
    try/except intact just as the parsing path does."""
    s = _cpython_strategy(dataflow_fusion=True)
    lines = ("@deco\ndef f():\n    pass\ntry:\n    g()\nfinally:\n    h()\n"
             "x = 1\n").splitlines()
    blocks = s._py_blocks_by_indent(lines)
    assert len(blocks) == 3, blocks
    assert blocks[0].startswith("@deco")
    assert blocks[1].startswith("try:") and "finally:" in blocks[1]


def test_stale_state_indices_are_not_reused_after_the_body_changes():
    """--pre-analysis caches state points as line indices into the seed's
    *original* text. Every language that hoists a preamble returns a
    different body from _state_prepare, so those indices no longer point
    where they did — CPython strips imports, Swift its `import` lines, Rust
    its crate attributes, Go its package clause.

    Reusing them put 52 of 116 four-segment children inside a compound
    statement.
    """
    s = _cpython_strategy(state_fusion=True)
    src = "import os\nimport sys\n\nx = 1\ny = 2\nz = 3\n"
    a = Seed(content=src, metadata={"segment_boundaries": [0, 1, 2],
                                    "most_complex_states": [{"line_idx": 1,
                                                             "category": "x",
                                                             "live_count": 1}]})
    b = Seed(content=src, metadata={})
    body, _donor, _ctx = s._state_prepare(a, b)
    # The prepared body is shorter than the original, so the cached indices
    # cannot apply to it.
    assert body != a.content
    random.seed(0)
    child = s.fuse(a, b)
    import ast as _ast
    _ast.parse(child.content)


def test_hang_prone_seeds_are_excluded():
    """A fused program that deadlocks does not fail, it burns the whole
    timeout — the most expensive failure there is, costing the valid rate
    and the throughput together."""
    import re
    import yaml
    with open(os.path.join(ROOT, "projects", "cpython", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    pats = [re.compile(e["pattern"]) for e in cfg["paths"]["seed_exclude_patterns"]]
    for probe in ("import threading\nlock = threading.Lock()\n",
                  "import asyncio\nasyncio.run(main())\n",
                  "from . import helper\n",
                  "print(__file__)\n"):
        assert any(p.search(probe) for p in pats), probe
    # And ordinary code is still kept.
    assert not any(p.search("x = 1\nprint(x)\n") for p in pats)


# ---------------------------------------------------------------------------
# The assembly pipeline — what made 82% of children invalid
# ---------------------------------------------------------------------------

def _strategy(**flags):
    from core.fusion import get_strategies
    return get_strategies("cpython", pre_analysis_enabled=True, **flags)[0]


def test_nested_imports_are_not_hoisted():
    """Hoisting an indented `import` empties the block it came from.

        try:
            import _md5
        except ImportError:

    became `try:` immediately followed by `except` — "expected an indented
    block after 'try' statement". This was the single largest cause of
    invalid children on the CPython Lib corpus, and it also turned a
    conditional import into a hard dependency.
    """
    s = _strategy(dataflow_fusion=True)
    src = ("import os\n"
           "try:\n"
           "    import _md5\n"
           "except ImportError:\n"
           "    _md5 = None\n")
    body, imports = s._extract_imports_and_body(src)
    assert imports == ["import os"]
    assert "import _md5" in body, "the conditional import must stay in its block"
    import ast as _ast
    _ast.parse(body)          # the try block still has a body


def test_parenthesised_imports_are_hoisted_whole():
    """A multi-line import must move as one statement. Matching only its
    first line leaves `    Foo,\\n)` behind, so both halves are syntax
    errors — 53 of ~70 remaining parse failures before this was fixed."""
    s = _strategy(dataflow_fusion=True)
    src = ("from typing import (\n"
           "    Any,\n"
           "    Optional,\n"
           ")\n"
           "x = 1\n")
    body, imports = s._extract_imports_and_body(src)
    assert len(imports) == 1 and "Optional" in imports[0]
    assert "Optional" not in body and body.strip() == "x = 1"


def test_toplevel_blocks_keep_decorators_and_clauses_together():
    """Splitting at "a line starting at column zero" tore apart every
    construct whose continuation is also at column zero: a decorator from
    its def, an `except` from its `try`."""
    s = _strategy(dataflow_fusion=True)
    src = ("@decorator\n"
           "def f():\n"
           "    pass\n"
           "try:\n"
           "    g()\n"
           "except ValueError:\n"
           "    pass\n")
    blocks = s._py_toplevel_blocks(src)
    assert len(blocks) == 2, blocks
    assert blocks[0].startswith("@decorator") and "def f" in blocks[0]
    assert blocks[1].startswith("try:") and "except" in blocks[1]
    import ast as _ast
    for b in blocks:
        _ast.parse(b)


def test_toplevel_blocks_fall_back_when_the_source_does_not_parse():
    """A child being re-fused may not parse; the splitter must degrade
    rather than raise."""
    s = _strategy(dataflow_fusion=True)
    assert s._py_toplevel_blocks("def (:\n") is not None


def test_state_fusion_does_not_reuse_indices_from_a_rewritten_body():
    """--pre-analysis caches state points as line indices into the seed's
    *original* text. CPython's _state_prepare strips module-level imports,
    so every removed line shifts the indices below it and a cut computed as
    "between two top-level statements" lands inside one.

    Measured: 52 of 116 four-segment children were invalid because of this.
    """
    from core.fusion import Seed
    s = _strategy(state_fusion=True)
    src = ("import os\nimport sys\n"
           "def f():\n    return 1\n"
           "def g():\n    return 2\n")
    host = Seed(content=src, metadata={
        "filename": "h",
        # Deliberately wrong for the stripped body: these index the original.
        "segment_boundaries": [0, 1, 3, 5],
        "most_complex_states": [{"line_idx": 5, "category": "x",
                                 "matched_text": "", "indent": "",
                                 "live_count": 9}],
    })
    body, _donor, _ctx = s._state_prepare(host, host)
    assert body != src, "this test is meaningless unless prepare rewrites the body"
    import random as _r
    _r.seed(0)
    for _ in range(10):
        child = s.fuse(host, host)
        import ast as _ast
        _ast.parse(child.content)     # would raise if a stale index were used


# ---------------------------------------------------------------------------
# False positives the first live run produced
# ---------------------------------------------------------------------------

def test_our_own_memory_cap_is_not_a_bug():
    """The driver caps memory through ASan's hard_rss_limit_mb. Hitting it
    makes ASan abort and CPython's fatal handler then prints "Fatal Python
    error: Aborted" — so checking the fatal pattern before the resource
    pattern files our own cap as an interpreter bug. It did, on the first
    live run."""
    out = ("......F...==54676==AddressSanitizer: hard rss limit exhausted "
           "(2048Mb vs 2118)\nFatal Python error: Aborted\n")
    v = analyzer.classify(out)
    assert v["is_bug"] is False and v["kind"] == "resource"


def test_allocator_diagnostic_quoted_in_an_assertion_is_not_a_bug():
    """CPython's own tests assert on the exact wording of allocator
    diagnostics, so their AssertionError messages contain the text
    verbatim. Matching it reports the test's *expectation* as a finding."""
    out = ('Traceback (most recent call last):\n'
           '  File "t.py", line 9, in test_pymem\n'
           'AssertionError: Regex didn\'t match: "Debug memory block at '
           'address p=(?:0x)?[0-9a-fA-F]+: API \'m\'"\n')
    assert analyzer.classify(out)["is_bug"] is False


def test_allocator_diagnostic_echoed_in_a_syntaxerror_is_not_a_bug():
    """A SyntaxError traceback echoes the offending source line. When that
    line happens to contain the diagnostic as a string literal, the oracle
    was reporting the source text CPython merely printed back."""
    out = ('  File "/tmp/x/b0b1cdcf.py", line 518\n'
           '    regex = (_testinternalcapi"Debug memory block at address '
           'p={ptr}: API \'m\'\\n"\n'
           '                              ^^^^^^^^^^^^\n'
           'SyntaxError: invalid syntax\n')
    assert analyzer.classify(out)["is_bug"] is False


def test_a_real_allocator_failure_is_still_reported():
    """The guard must not swallow the genuine article, which the
    interpreter prints at column zero on its own line."""
    out = ("Debug memory block at address p=0x5555: API 'o'\n"
           "    bad trailing pad byte\n"
           "Fatal Python error: _PyMem_DebugRawFree: bad trailing pad byte\n")
    v = analyzer.classify(out)
    assert v["is_bug"] is True and v["kind"] == "fatal"


def test_signal_names_must_stand_alone_on_a_line():
    """`ConnectionAbortedError` contains "Aborted", and any program that
    enumerates the exception hierarchy prints it — inside a string literal.
    A bare substring match reported that as a crash. The shell and the
    runtime always print a signal on its own line."""
    hierarchy = ("  _pep_map = '''\\n +-- ConnectionAbortedError EPIPE\\n'''\\n"
                 "Ran 1 test in 0.001s\\n\\nOK\\n")
    assert analyzer.classify(hierarchy)["is_bug"] is False
    for real in ("Segmentation fault\n", "Segmentation fault (core dumped)\n",
                 "Aborted\n", "Bus error (core dumped)\n"):
        v = analyzer.classify(real)
        assert v["is_bug"] is True and v["kind"] == "signal", real


def test_classify_stays_linear_on_very_long_lines():
    """The assertion patterns were unanchored, and `[\\w./+-]+\\.[ch]` has to
    try every starting offset and every split of a long run before failing.
    Fused output routinely carries lines tens of kilobytes long — a large
    repr, an `-X importtime` dump, a giant traceback — and one search over
    such a line pegged a worker at 100% CPU *holding the GIL*, starving the
    other fifteen and wedging the campaign for half an hour at a time.

    The cumulative throughput counter kept reading 14.7 tests/s throughout,
    which is why this is pinned by a timing assertion rather than by eye.
    """
    import time
    hostile = "x" + ("/a.b_c-d" * 25000) + "\n"      # ~200 KB, no .c/.h
    start = time.time()
    analyzer.classify(hostile)
    elapsed = time.time() - start
    assert elapsed < 1.0, f"classify took {elapsed:.1f}s on a 200 KB line"


def test_assertion_prefix_may_be_a_full_path():
    """glibc prints argv[0], which is often an absolute path, so the
    optional program prefix has to allow slashes — while staying bounded,
    or the anchoring above buys nothing."""
    bare = "python: Objects/dictobject.c:1: f: Assertion `x' failed.\n"
    full = "/build/python: Objects/dictobject.c:1: f: Assertion `x' failed.\n"
    assert analyzer.crash_signature(bare) == analyzer.crash_signature(full)
    assert analyzer.crash_signature(bare) is not None
