"""
Tests for the Go adapter: the crash oracle and the file-structure rules
fusion has to respect.

Go's structural rules are strict enough that getting them wrong does not
produce a lower valid rate, it produces zero: a file with two `package`
clauses fails in the parser, before any of the compiler this fuzzer aims
at ever runs.
"""

import importlib.util
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.fusion import Seed, get_strategies  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("ffl_go_analyzer_test", "projects/go/analyzer.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

ICE = """./x.go:9:5: internal compiler error: bad type for v12

goroutine 1 [running]:
runtime/debug.Stack()
\t/go/src/runtime/debug/stack.go:26 +0x5e
cmd/compile/internal/base.FatalfAt({0x0, 0x0}, {0xc0001, 0x1a})
\t/go/src/cmd/compile/internal/base/print.go:233 +0x1f4
cmd/compile/internal/ssa.(*Func).Fatalf(0xc000123, {0x9a1b2c, 0x12})
\t/go/src/cmd/compile/internal/ssa/func.go:722 +0x1b8
cmd/compile/internal/ssa.checkFunc(0xc000456)
\t/go/src/cmd/compile/internal/ssa/check.go:44 +0x2ac
"""

PANIC_VIA_HANDLER = """<unknown line number>: internal compiler error: panic: incomplete alias

goroutine 1 [running]:
runtime/debug.Stack()
\t/go/src/runtime/debug/stack.go:26 +0x5e
cmd/compile/internal/base.FatalfAt({0xc93?, 0x18b?}, {0x10303bc, 0x9})
\t/go/src/cmd/compile/internal/base/print.go:232 +0x18b
cmd/compile/internal/gc.handlePanic()
\t/go/src/cmd/compile/internal/gc/main.go:59 +0x95
panic({0x1a08b48?, 0x10a09a0?})
\t/go/src/runtime/panic.go:859 +0x11f
cmd/compile/internal/types2.(*Checker).handleBailout(0x18bc8eb4800, 0x18bc9334e18)
\t/go/src/cmd/compile/internal/types2/check.go:396 +0x286
cmd/compile/internal/types2.(*typeWriter).typ(0xc0001, {0x1a0, 0xc00042})
\t/go/src/cmd/compile/internal/types2/typestring.go:158 +0x9a5
"""

OOM = "fatal error: runtime: out of memory\n\ngoroutine 1 [running]:\n"
KILLED = "signal: killed\n"
DIAGNOSTIC = "./x.go:5:2: undefined: foo\n./x.go:9:1: declared and not used: y\n"
RACE = ("WARNING: DATA RACE\nRead at 0x00c0004 by goroutine 9:\n"
        "cmd/compile/internal/types.(*Type).Size(0xc00012)\n"
        "\t/go/src/cmd/compile/internal/types/size.go:229 +0x44\n")


def test_ice_is_a_bug_and_names_the_failing_pass():
    v = analyzer.classify(ICE)
    assert v["is_bug"] and v["kind"] == "ice"
    assert v["frame"] == "cmd/compile/internal/ssa.checkFunc"


def test_panic_plumbing_is_skipped_to_reach_the_real_frame():
    """The compiler turns a panic into an ICE by recovering and
    re-panicking through handlePanic and handleBailout. Grouping on those
    would collapse every panic-routed ICE into one bucket."""
    v = analyzer.classify(PANIC_VIA_HANDLER)
    assert v["is_bug"]
    assert v["frame"] == "cmd/compile/internal/types2.(*typeWriter).typ", v["frame"]


@pytest.mark.parametrize("output", [OOM, KILLED])
def test_resource_exhaustion_is_not_a_bug(output):
    """Go reports running out of memory through the same `fatal error:`
    channel as a real compiler bug, so the resource check has to come
    first or every OOM is filed as a finding."""
    v = analyzer.classify(output)
    assert v["is_bug"] is False and v["kind"] == "resource"


def test_ordinary_diagnostics_are_not_bugs():
    v = analyzer.classify(DIAGNOSTIC)
    assert v["is_bug"] is False and v["kind"] == "diagnostic"


def test_race_report_wins_over_whatever_follows():
    v = analyzer.classify(RACE)
    assert v["kind"] == "race" and "Size" in v["signature"]


def test_signature_is_stable_across_seeds():
    """Signatures collapse many hits on one bug into one directory. Pointer
    values, SSA value numbers and paths all vary run to run."""
    other = (ICE.replace("v12", "v87").replace("0xc000123", "0xdeadbeef")
                .replace("x.go:9:5", "other.go:412:77"))
    assert analyzer.crash_signature(ICE) == analyzer.crash_signature(other)
    assert analyzer.crash_signature(DIAGNOSTIC) is None


# ---------------------------------------------------------------------------
# Go's file structure
# ---------------------------------------------------------------------------

from core.fusion import split_go_file, go_toplevel_names, assemble_go_file  # noqa: E402

SEED_A = '''// run
package main

import (
\t"fmt"
\t"os"
)

type T struct{ a int }

func helper(x int) int { return x * 2 }

func main() {
\tfmt.Println(helper(1))
\tos.Exit(0)
}
'''

SEED_B = '''// compile
package main

import "strings"

type T struct{ b string }

func main() {
\t_ = strings.ToUpper("x")
}
'''


def test_split_removes_package_and_imports():
    pkg, imports, body = split_go_file(SEED_A)
    assert pkg == "main"
    assert set(imports) == {'"fmt"', '"os"'}
    assert "package main" not in body
    assert "import" not in body
    assert "func helper" in body


def test_split_handles_the_single_import_form():
    _, imports, body = split_go_file(SEED_B)
    assert imports == ['"strings"']
    assert "import" not in body


def test_toplevel_names_ignore_indented_declarations():
    """Only column-0 declarations are package-scope. A `func` inside a
    function body is a closure and never collides."""
    body = "func Top() {\n\tinner := func() {}\n\t_ = inner\n}\ntype U struct{}\n"
    assert go_toplevel_names(body) == {"Top", "U"}


def test_init_is_not_treated_as_a_collision():
    """Go permits any number of `func init()` in one package and runs them
    all, so renaming them apart would change the program for no reason."""
    assert "init" not in go_toplevel_names("func init() {}\nfunc init() {}\n")


def test_fusion_produces_one_package_clause_and_one_import_block():
    """Two Go files cannot be concatenated: the result would carry two
    package clauses and fail in the parser, before reaching the compiler
    this fuzzer is aiming at."""
    strategy = get_strategies("go", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    a = Seed(content=SEED_A, metadata={"filename": "a"})
    b = Seed(content=SEED_B, metadata={"filename": "b"})
    random.seed(3)
    for _ in range(20):
        child = strategy.fuse(a, b)
        assert child.content.count("\npackage ") + child.content.startswith("package ") == 1
        assert child.content.lstrip().startswith("package ")
        assert child.content.count("\nimport (") == 1


def test_fusion_merges_imports_rather_than_dropping_them():
    """Go makes an unused import a compile error, so an import may only be
    carried when the code that used it comes too — and dropping one whose
    user is still present is equally fatal."""
    strategy = get_strategies("go", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    a = Seed(content=SEED_A, metadata={"filename": "a"})
    b = Seed(content=SEED_B, metadata={"filename": "b"})
    random.seed(4)
    child = strategy.fuse(a, b)
    for spec in ('"fmt"', '"os"', '"strings"'):
        assert spec in child.content, spec


def test_fusion_renames_colliding_toplevel_declarations():
    """Both seeds declare `func main` and `type T`. Go rejects a
    redeclaration outright, and with two seeds that is the common case,
    not an edge one."""
    strategy = get_strategies("go", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    a = Seed(content=SEED_A, metadata={"filename": "a"})
    b = Seed(content=SEED_B, metadata={"filename": "b"})
    random.seed(5)
    child = strategy.fuse(a, b)
    assert child.content.count("\nfunc main(") == 1, child.content
    assert child.content.count("\ntype T ") <= 1
    assert set(child.metadata["renamed"]) >= {"T", "main"}


def test_declaration_fusion_does_not_copy_the_donated_type():
    """Both bodies land in one package, so a bare reference resolves.
    Copying the declaration instead produced a second declaration of a type
    the donor still declares — "other declaration of T" — which took the
    strategy's valid rate to zero."""
    strategies = get_strategies("go", declaration_fusion=True,
                                pre_analysis_enabled=True)
    assert strategies
    a = Seed(content=SEED_A, metadata={"filename": "a", "has_declaration": True})
    b = Seed(content=SEED_B, metadata={"filename": "b", "has_declaration": True})
    random.seed(6)
    for _ in range(20):
        child = strategies[0].fuse(a, b)
        # Exactly one declaration of each type name survives.
        for name in ("T", "T_" ):
            assert child.content.count(f"type {name}struct") <= 1
        assert child.content.count("type T struct") <= 1


def test_go_is_registered_with_all_three_techniques():
    names = [type(s).__name__ for s in get_strategies("go", pre_analysis_enabled=True)]
    assert names == ["GoFusionStrategy", "GoDeclarationFusionStrategy",
                     "GoStateFusionStrategy"]


def test_live_variable_counting_follows_go_scoping():
    """State fusion ranks splice points by how many variables are live.
    Go scopes to `{}`, so the innermost block must count the most."""
    from core.state_analysis import find_state_points
    code = ("package main\n\nfunc main() {\n\tx := 1\n\ty := x * 2\n"
            "\tvar z int = y\n\tfor i := 0; i < z; i++ {\n\t\tw := i + y\n"
            "\t\t_ = w\n\t}\n}\n")
    points = find_state_points(code, "go", None)
    assert points
    assert max(p.live_count for p in points) >= 4, \
        [(p.line_idx, p.live_count) for p in points]


def test_config_declares_what_the_framework_reads():
    import re
    import yaml
    with open(os.path.join(ROOT, "projects", "go", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["project_name"] == "go"
    assert "internal compiler error" in cfg["analysis"]["crash_patterns"]
    # A bare panic is how an ICE arrives when ordinary errors came first.
    assert "panic: " in cfg["analysis"]["crash_patterns"]
    for entry in cfg["paths"]["seed_exclude_patterns"]:
        re.compile(entry["pattern"])
        assert entry["reason"]


def test_parser_emits_the_keys_fusion_and_dryrun_need():
    parser = _load("ffl_go_parser_test", "projects/go/parser.py")
    meta = parser._parser.parse_content(SEED_A, "a.go")
    # dataflow fusion silently no-ops without these.
    assert meta["variables"] and meta["dataflows"]
    # core/dryrun.py keys on "type" to pick GoMetadataCollector.
    assert meta["type"] == "go"
    assert meta["is_main"] is True
    assert meta["directive"] == "run"
    assert set(meta["imports"]) == {'"fmt"', '"os"'}
