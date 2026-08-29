"""
Cross-language smoke test: every fusion strategy every project exposes must
fuse a trivial pair without raising.

This exists because a NameError inside a strategy is invisible in
production: core/orchestrator.py's _fuse_once catches Exception, logs
"Fusion error" and skips the iteration, so a broken strategy looks like a
quiet fuzzing run rather than a crash. A one-line import edit in one
language's strategy silently killed two others' state fusion exactly that
way.

These assertions are deliberately weak on content — per-strategy semantics
are tested elsewhere. All this pins is "it runs for every project".
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.fusion import Seed, get_strategies  # noqa: E402

# One minimal, valid-ish seed template per project. `%d` slots are filled
# with the seed index so the two parents differ.
SEED_TEMPLATES = {
    "swift": "func f%d(a: Int) -> Int {\n    let x = a + %d\n    let y = x * 2\n    return y\n}\n",
    "clang": "int f%d(int a) {\n    int x = a + %d;\n    int y = x * 2;\n    return y;\n}\n",
    # GCC reuses clang's three strategies (core/fusion.py's registry), but
    # runs them against its own corpus and parser, so it gets its own row.
    "gcc": "int g%d(int a) {\n    int x = a + %d;\n    int y = x * 2;\n    return y;\n}\n",
    "cpython": "def f%d(a):\n    x = a + %d\n    y = x * 2\n    return y\n",
    "php": "--TEST--\nt%d\n--FILE--\n<?php\n$x = %d;\n$y = $x * 2;\nvar_dump($y);\n?>\n",
    "rust": "fn f%d(a: i32) -> i32 {\n    let x = a + %d;\n    let y = x * 2;\n    y\n}\n",
    "go": ("package main\n\nimport \"fmt\"\n\nfunc f%d(a int) int {\n"
           "\tx := a + %d\n\ty := x * 2\n\treturn y\n}\n\n"
           "func main() { fmt.Println(f%d(1)) }\n"),
    "haskell": "f%d :: Int -> Int\nf%d a = let x = a + %d in x * 2\n",
    "flang": "program p%d\n  integer :: x\n  x = %d\n  print *, x\nend program\n",
    "mlir": ("module {\n  func.func @f%d(%%a: i32) -> i32 {\n"
             "    %%c = arith.constant %d : i32\n"
             "    %%0 = arith.addi %%a, %%c : i32\n    return %%0 : i32\n  }\n}\n"),
}

TECHNIQUES = [
    ("dataflow", {"dataflow_fusion": True}),
    ("state", {"state_fusion": True}),
    ("declaration", {"declaration_fusion": True}),
    ("struct", {"struct_fusion": True}),
]

CASES = [(project, technique, flags)
         for project in SEED_TEMPLATES
         for technique, flags in TECHNIQUES]


def _parents(project):
    template = SEED_TEMPLATES[project]
    slots = template.count("%d")
    return [Seed(content=template % tuple([i] * slots), metadata={"filename": f"s{i}"})
            for i in range(2)]


@pytest.mark.parametrize("pre_analysis", [True, False])
@pytest.mark.parametrize("project,technique,flags", CASES,
                         ids=[f"{p}-{t}" for p, t, _ in CASES])
def test_strategy_fuses_without_raising(project, technique, flags, pre_analysis):
    random.seed(4)
    strategies = get_strategies(project, pre_analysis_enabled=pre_analysis, **flags)
    if not strategies:
        pytest.skip(f"{project} does not support --{technique}-fusion")

    a, b = _parents(project)
    for strategy in strategies:
        for _ in range(3):
            child = strategy.fuse(a, b)
            assert child.content is not None
            if hasattr(strategy, "fuse_bidirectional"):
                children = strategy.fuse_bidirectional(a, b)
                assert children and all(c.content is not None for c in children)


# ---------------------------------------------------------------------------
# get_strategies' registry contract
# ---------------------------------------------------------------------------

def test_struct_fusion_flag_is_rust_only():
    """--struct-fusion is Rust's own name for its declaration technique.
    Turning it into a generic alias (which a registry rewrite did, silently)
    enables declaration fusion for nine languages the user never named."""
    from core.fusion import get_strategies as _gs
    assert [type(s).__name__ for s in _gs("rust", struct_fusion=True, pre_analysis_enabled=True)] \
        == ["RustStructFusionStrategy"]
    for project in ("clang", "php", "cpython", "mlir", "swift", "flang", "haskell", "naga"):
        assert _gs(project, struct_fusion=True, pre_analysis_enabled=True) == [], \
            f"--struct-fusion must be a no-op for {project}"


def test_declaration_flag_reaches_rusts_struct_strategy():
    """The reverse alias does hold: --declaration-fusion is the portable
    name and must work for Rust too."""
    from core.fusion import get_strategies as _gs
    assert [type(s).__name__ for s in _gs("rust", declaration_fusion=True, pre_analysis_enabled=True)] \
        == ["RustStructFusionStrategy"]


def test_no_flags_enables_every_technique_the_project_has():
    from core.fusion import get_strategies as _gs
    assert len(_gs("clang", pre_analysis_enabled=True)) == 3      # dataflow+declaration+state
    # Rust gained state fusion when the adapter was rebuilt: the pieces
    # were already there (a brace-mode LIVE_VAR_CONFIGS entry, `let` as the
    # declaration form), only the four template hooks were missing.
    assert len(_gs("rust", pre_analysis_enabled=True)) == 3
    assert len(_gs("go", pre_analysis_enabled=True)) == 3


def test_unknown_project_gets_no_strategies():
    """main.py fails fast on an empty pool; an unknown name must not fall
    through to some other language's strategies."""
    from core.fusion import get_strategies as _gs
    assert _gs("no_such_project", pre_analysis_enabled=True) == []


def test_every_registry_entry_is_constructible():
    from core.fusion import STRATEGY_REGISTRY, get_strategies as _gs
    for project in STRATEGY_REGISTRY:
        built = _gs(project, pre_analysis_enabled=True)
        assert built, f"{project} produced no strategies"
        for s in built:
            assert hasattr(s, "fuse") and hasattr(s, "is_viable_pair")


# ---------------------------------------------------------------------------
# Dataflow fusion is one technique in every language: rename a name in B to
# a name from A. Nothing is inserted, nothing is declared.
# ---------------------------------------------------------------------------

DATAFLOW_PROJECTS = sorted(SEED_TEMPLATES)


@pytest.mark.parametrize("project", DATAFLOW_PROJECTS)
def test_dataflow_is_a_rename_and_never_inserts(project):
    """rename_across must leave A untouched and only rewrite B.

    Each language used to synthesise its own bridge declaration — a
    `static long fusion_var` for C, an `IORef` for Haskell, a whole
    `func.func @_ffl_bridge_N` for MLIR. Those are gone; a regression that
    reintroduced one would show up here as A changing.
    """
    strategies = get_strategies(project, dataflow_fusion=True,
                                pre_analysis_enabled=True)
    if not strategies:
        pytest.skip(f"{project} has no dataflow strategy")
    strategy = strategies[0]
    a, b = _parents(project)

    random.seed(5)
    for _ in range(30):
        out_a, out_b = strategy.rename_across(a.content, b.content)
        assert out_a == a.content, "dataflow fusion must not modify A"
        assert len(out_b.splitlines()) == len(b.content.splitlines()), \
            "a rename cannot change B's line count"


@pytest.mark.parametrize("project", DATAFLOW_PROJECTS)
def test_dataflow_name_pool_excludes_keywords(project):
    """Renaming `func`/`def`/`fn` rewrites every declaration at once, which
    fails deterministically instead of by chance — so those tokens must not
    be in the pool even though invalid children are otherwise fine."""
    strategies = get_strategies(project, dataflow_fusion=True,
                                pre_analysis_enabled=True)
    if not strategies:
        pytest.skip(f"{project} has no dataflow strategy")
    a, _ = _parents(project)
    names = strategies[0]._dataflow_names(a.content)
    forbidden = {"def", "func", "fn", "class", "return", "let", "var",
                 "program", "module", "import", "where"}
    assert not (set(names) & forbidden), f"{project} pool contains keywords: {names}"


def test_renaming_a_name_to_itself_is_a_noop():
    """Otherwise a whole fusion attempt is spent rewriting a name to what it
    already was, and the pair looks fused when nothing happened."""
    strategy = get_strategies("clang", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    code = "int f(int shared) { return shared; }\n"
    assert strategy.rename_across(code, code, ["shared"], ["shared"]) == (code, code)


# ---------------------------------------------------------------------------
# Renamed lines carry a trailing "dataflow fusion" comment
# ---------------------------------------------------------------------------

COMMENT_TOKEN = {
    "clang": "//", "gcc": "//", "php": "//", "rust": "//", "swift": "//",
    "mlir": "//", "cpython": "#", "haskell": "--", "flang": "!",
    "go": "//",
}


@pytest.mark.parametrize("project", DATAFLOW_PROJECTS)
def test_renamed_line_is_tagged_in_the_projects_own_comment_syntax(project):
    """The tag has to be a comment in the target language, or it is a syntax
    error rather than a marker — Haskell needs `--`, Fortran `!`, Python `#`."""
    strategies = get_strategies(project, dataflow_fusion=True,
                                pre_analysis_enabled=True)
    if not strategies:
        pytest.skip(f"{project} has no dataflow strategy")
    strategy = strategies[0]
    a, b = _parents(project)

    random.seed(2)
    for _ in range(60):
        _, out_b = strategy.rename_across(a.content, b.content)
        tagged = [ln for ln in out_b.splitlines() if "dataflow fusion" in ln]
        if tagged:
            token = COMMENT_TOKEN[project]
            assert all(f"{token} dataflow fusion" in ln for ln in tagged), tagged
            # Same line as the code it marks, so a line-based reducer cannot
            # drop the tag while keeping the statement.
            assert all(ln.strip() != f"{token} dataflow fusion" for ln in tagged)
            return
    pytest.skip(f"{project}'s template pair never produced a rename")


def test_only_changed_lines_are_tagged():
    strategy = get_strategies("clang", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    before = "int a = 1;\nint b = 2;\nint c = 3;\n"
    after = "int a = 1;\nint z = 2;\nint c = 3;\n"
    out = strategy._tag_renamed_lines(before, after).splitlines()
    assert out[0] == "int a = 1;"
    assert out[1] == "int z = 2;  // dataflow fusion"
    assert out[2] == "int c = 3;"


def test_line_continuation_is_never_tagged():
    """A C macro continues on the next line only if the backslash is the last
    character. Appending a comment after it ends the macro early, so a
    multi-line macro would silently lose its body."""
    strategy = get_strategies("clang", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    before = "#define M(x) \\\n    foo(x)\n"
    after = "#define M(y) \\\n    foo(y)\n"
    out = strategy._tag_renamed_lines(before, after)
    assert out.splitlines()[0].endswith("\\"), out
    assert out.splitlines()[1].endswith("// dataflow fusion")


def test_tagging_preserves_the_trailing_newline():
    """splitlines() drops it; putting it back matters because a file that
    stops without a newline can change how the last line is diagnosed."""
    strategy = get_strategies("clang", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    assert strategy._tag_renamed_lines("int a;\n", "int b;\n").endswith("\n")
    assert not strategy._tag_renamed_lines("int a;", "int b;").endswith("\n")


# ---------------------------------------------------------------------------
# All three techniques tag what they changed
# ---------------------------------------------------------------------------

def _fuse_until_tagged(strategy, a, b, kind, tries=80):
    """Fuse repeatedly until a child carries `kind`'s tag; return that line.

    Fusion is randomised and every technique legitimately no-ops sometimes
    (renaming a name to itself, a donor with nothing to donate), so a single
    fuse proves nothing either way.
    """
    for i in range(tries):
        random.seed(i)
        child = strategy.fuse(a, b)
        for line in (child.content or "").splitlines():
            if f"{kind} fusion" in line:
                return line
    return None


def test_state_fusion_tags_its_splice_point():
    for project in ("clang", "cpython", "php", "swift", "mlir", "haskell", "flang"):
        strategies = get_strategies(project, state_fusion=True,
                                    pre_analysis_enabled=True)
        assert strategies, project
        a, b = _parents(project)
        line = _fuse_until_tagged(strategies[0], a, b, "state")
        assert line, f"{project} state fusion never tagged its splice"
        assert COMMENT_TOKEN[project] in line, line


def test_cpython_dataflow_tags_although_it_bypasses_rename_across():
    """CPython substitutes through replace_one_b_occurrence (it needs
    indentation-aware replacement) rather than rename_across, so it has to
    tag the changed lines itself — a gap when the rename was unified."""
    strategy = get_strategies("cpython", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    # a_var comes from collect_top_level_assigned_vars, so A needs a
    # top-level binding; a function-only seed gives the technique nothing.
    a = Seed(content="alpha = 1\n\ndef f(z):\n    return z + alpha\n",
             metadata={"filename": "a"})
    b = Seed(content="beta = 2\n\ndef g(z):\n    return z + beta\n",
             metadata={"filename": "b"})
    line = _fuse_until_tagged(strategy, a, b, "dataflow")
    assert line and line.rstrip().endswith("# dataflow fusion"), line


def test_rust_dataflow_tags_the_rebound_statement():
    """Rust rebinds one statement inside a merged stream rather than
    rewriting a whole body, so it tags that statement directly instead of
    going through the line diff."""
    strategy = get_strategies("rust", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    # _LET_TYPE_RE only sees annotated lets, and only main's body is
    # flattened into the shared statement stream.
    tpl = ("fn main() {\n    let b%d: i32 = %d;\n"
           "    let c%d: i32 = b%d * 2;\n    println!(\"{}\", c%d);\n}\n")
    a, b = [Seed(content=tpl % ((i,) * 5), metadata={"filename": f"s{i}"})
            for i in (1, 2)]
    line = _fuse_until_tagged(strategy, a, b, "dataflow")
    assert line and line.rstrip().endswith("// dataflow fusion"), line


def test_declaration_fusion_tags_the_injected_declaration():
    """Each language injects a different thing — a struct/class for C, a
    typeclass superclass constraint for Haskell — so each needs a donor that
    actually declares one."""
    cases = {
        "clang": "struct S%d { int a; };\nint g%d(int x) { return x; }\n",
        "cpython": "class C%d:\n    pass\n\ndef f%d(z):\n    return z\n",
        "haskell": ("class Shape%d a where\n  area%d :: a -> Double\n\n"
                    "f%d :: (Show b) => b -> String\nf%d x = show x\n"),
    }
    for project, tpl in cases.items():
        strategies = get_strategies(project, declaration_fusion=True,
                                    pre_analysis_enabled=True)
        assert strategies, project
        n = tpl.count("%d")
        a, b = [Seed(content=tpl % ((i,) * n),
                     metadata={"filename": f"s{i}", "has_declaration": True})
                for i in (1, 2)]
        line = _fuse_until_tagged(strategies[0], a, b, "declaration")
        assert line, f"{project} declaration fusion never tagged its injection"
        assert COMMENT_TOKEN[project] in line, line
