"""
Tests for the GCC adapter's oracle (projects/gcc/analyzer.py).

The oracle is the part of a new adapter that decides whether a run produces
findings or noise, and it is the part that cannot be checked by running the
fuzzer: a wrong "is_bug" simply yields a directory of results nobody reads.
The strings below are real GCC output shapes.
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


analyzer = _load("ffl_gcc_analyzer_test", "projects/gcc/analyzer.py")


# ---------------------------------------------------------------------------
# What counts as a bug
# ---------------------------------------------------------------------------

ICE_WITH_LOCATION = """\
t.C: In instantiation of 'void f(T) [with T = int]':
t.C:9:5: internal compiler error: in tsubst_expr, at cp/pt.cc:21203
    9 |     f<int>(0);
      |     ^~~~~~~~~
0x1e3f4a2 internal_error(char const*, ...)
Please submit a full bug report, with preprocessed source.
"""

ICE_DURING_PASS = """\
during GIMPLE pass: vrp
t.c: In function 'main':
t.c:12:1: internal compiler error: Segmentation fault
0xabc123 crash_signal
Please submit a full bug report.
"""

CHECKING_FAILURE = """\
t.c: In function 'g':
t.c:4:1: error: invalid PHI argument
t.c:4:1: internal compiler error: verify_ssa failed
"""

# The compiler died, but for lack of memory, not for a reason a GCC
# maintainer can act on.
OOM_KILLED = "gcc: internal compiler error: Killed (program cc1plus)\n"
ULIMIT_OOM = "cc1plus: out of memory allocating 268435456 bytes after a total of 4194304 bytes\n"
VM_EXHAUSTED = "virtual memory exhausted: Cannot allocate memory\n"

# The overwhelmingly common outcome for a fused program.
ORDINARY_ERROR = "t.c:3:12: error: expected ';' before '}' token\n"
CLEAN = ""


@pytest.mark.parametrize("output,kind", [
    (ICE_WITH_LOCATION, "ice"),
    (ICE_DURING_PASS, "ice"),
    (CHECKING_FAILURE, "checking"),
])
def test_compiler_failures_are_bugs(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is True
    assert verdict["kind"] == kind
    assert verdict["signature"]


@pytest.mark.parametrize("output", [OOM_KILLED, ULIMIT_OOM, VM_EXHAUSTED])
def test_resource_exhaustion_is_not_a_bug(output):
    """These match config.yaml's crash_patterns ("internal compiler error",
    "out of memory") but say nothing about GCC — the same input compiles on
    a bigger machine. Reporting them buries the real findings."""
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False
    assert verdict["kind"] == "resource"
    assert verdict["signature"] is None


@pytest.mark.parametrize("output,kind", [(ORDINARY_ERROR, "diagnostic"), (CLEAN, "clean")])
def test_ordinary_outcomes_are_not_bugs(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False
    assert verdict["kind"] == kind


def test_sanitizer_report_wins_over_the_ice_line_that_follows_it():
    """With FFL_GCC_SANITIZE=1 an ASan report names the exact bad access
    inside the compiler, which is strictly more actionable than the generic
    ICE line GCC's signal handler prints afterwards."""
    output = (
        "ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0\n"
        "    #0 0x7f in gimple_call_arg gimple.h:3312\n"
        "SUMMARY: AddressSanitizer: heap-use-after-free gimple.h:3312 in gimple_call_arg\n"
        "t.c:1:1: internal compiler error: Segmentation fault\n"
    )
    verdict = analyzer.classify(output)
    assert verdict["kind"] == "asan"
    assert "heap-use-after-free" in verdict["signature"]


# ---------------------------------------------------------------------------
# Signature stability — the deduplication key
# ---------------------------------------------------------------------------

def test_same_invariant_from_different_seeds_gets_one_signature():
    """Signatures are what collapse a thousand hits on one broken invariant
    into one entry in outputs/. They must not carry per-seed detail."""
    a = "a_seed_9812.C:9:5: internal compiler error: in tsubst_expr, at cp/pt.cc:21203\n"
    b = "other_1.C:412:77: internal compiler error: in tsubst_expr, at cp/pt.cc:21203\n"
    assert analyzer.crash_signature(a) == analyzer.crash_signature(b)


def test_same_invariant_from_different_passes_gets_different_signatures():
    """The same assertion tripped from two different passes is two bugs, so
    the failing pass belongs in the key."""
    vrp = "during GIMPLE pass: vrp\nt.c:1:1: internal compiler error: verify_gimple failed\n"
    dom = "during GIMPLE pass: dom\nt.c:1:1: internal compiler error: verify_gimple failed\n"
    assert analyzer.crash_signature(vrp) != analyzer.crash_signature(dom)


def test_non_bugs_have_no_signature():
    for output in (ORDINARY_ERROR, OOM_KILLED, CLEAN):
        assert analyzer.crash_signature(output) is None


# ---------------------------------------------------------------------------
# Seed analysis — what determines how a seed is compiled
# ---------------------------------------------------------------------------

def test_cxx_content_in_a_dot_c_file_still_goes_to_gxx():
    """Fusion mixes corpora, so a child can inherit C++ from one parent and
    a .c extension from the other. Sending it to `gcc` would reject it in
    the front end, wasting the execution."""
    assert analyzer.analyze_seed("#include <vector>\nstd::vector<int> v;\n", ".c")["is_cxx"]
    assert analyzer.analyze_seed("template <class T> T f(T x) { return x; }\n", ".c")["is_cxx"]
    assert not analyzer.analyze_seed("int main(void) { return 0; }\n", ".c")["is_cxx"]


def test_dejagnu_options_are_honoured_but_filtered():
    """A vectoriser test compiled without its dg-options exercises nothing.
    But -m<arch> and -I paths would fail in the driver rather than in the
    compiler, so they are dropped."""
    src = '/* { dg-options "-O3 -ftree-vectorize -mavx512f -I../include" } */\n'
    facts = analyzer.analyze_seed(src, ".c")
    assert facts["dg_options"] == ["-O3", "-ftree-vectorize"]
    assert facts["is_dejagnu"] is True


def test_seed_without_directives_yields_no_options():
    facts = analyzer.analyze_seed("int main(void) { return 0; }\n", ".c")
    assert facts["dg_options"] == []
    assert facts["is_dejagnu"] is False
    assert facts["dg_do"] is None


# ---------------------------------------------------------------------------
# Wiring: the pieces the framework loads by name
# ---------------------------------------------------------------------------

def test_gcc_is_registered_with_all_three_techniques():
    from core.fusion import get_strategies
    names = [type(s).__name__ for s in get_strategies("gcc", pre_analysis_enabled=True)]
    assert names == ["ClangFusionStrategy", "ClangDeclarationFusionStrategy",
                     "ClangStateFusionStrategy"]


def test_parser_emits_the_dataflow_keys_fusion_requires():
    """ClangFusionStrategy.interleave_code_blocks returns both parents
    untouched on `if not dataflow1 or not dataflow2`, so a parser that omits
    these keys makes dataflow fusion a silent no-op."""
    parser = _load("ffl_gcc_parser_test", "projects/gcc/parser.py")
    meta = parser._parser.parse_content(
        "int main(void) {\n  int x = 1;\n  int y = x + 2;\n  return y;\n}\n", "t.c")
    assert meta["variables"]
    assert meta["dataflows"] and all(isinstance(g, list) for g in meta["dataflows"])
    assert meta["type"] == "c"
    assert parser._parser.parse_content("class A {};\n", "t.cpp")["type"] == "cpp"


def test_config_declares_what_the_framework_reads():
    import yaml
    with open(os.path.join(ROOT, "projects", "gcc", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["project_name"] == "gcc"
    assert "internal compiler error" in cfg["analysis"]["crash_patterns"]
    assert cfg["paths"]["seed_corpus"].endswith("corpus.db")
    # Every exclusion pattern must compile, or corpus loading dies at startup.
    import re
    for entry in cfg["paths"]["seed_exclude_patterns"]:
        re.compile(entry["pattern"])
        assert entry["reason"]
