"""
Tests for the Triton adapter: the crash oracle, the RUN-line parsing, and
the section splitting the corpus depends on.

triton-opt is an MLIR pass driver — a compiler tool, not an executor — so
the expected outcome of a malformed input is a diagnostic, not a crash. A
fused MLIR module is usually ill-typed (two modules' tensor layouts rarely
agree), which makes ordinary diagnostics the common case and the oracle's
real job separating the assertion failures from that flood.
"""

import importlib.util
import os
import re
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


analyzer = _load("ffl_triton_analyzer_test", "projects/triton/analyzer.py")
setup_mod = _load("ffl_triton_setup_test", "projects/triton/setup.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

ASSERT = ("triton-opt: /src/lib/Dialect/TritonGPU/IR/Dialect.cpp:1234: "
          "void anonymous::verify(): Assertion `rank == 2' failed.\n"
          "Aborted (core dumped)\n")
UNREACHABLE = "UNREACHABLE executed at /src/lib/Conversion/Foo.cpp:99!\n"
LLVM_ERROR = "LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.foo\n"
CRASH = ("PLEASE submit a bug report to https://github.com/llvm/llvm-project\n"
         "Stack dump:\n"
         "0.\tProgram arguments: triton-opt x.mlir\n"
         " #0 0x0000561 llvm::sys::PrintStackTrace(llvm::raw_ostream&) + 121\n"
         " #1 0x0000562 mlir::triton::gpu::TritonGPUCoalescePass::runOnOperation() + 44\n")

# Not findings.
DIAGNOSTIC = ("x.mlir:12:34: error: 'ttg.convert_layout' op requires the same "
              "encoding\n    %0 = ttg.convert_layout %arg0\n")
OOM = "LLVM ERROR: out of memory\n"
BAD_ALLOC = "std::bad_alloc\n"


@pytest.mark.parametrize("output,kind", [
    (ASSERT, "assert"),
    (UNREACHABLE, "unreachable"),
    (LLVM_ERROR, "llvm_error"),
    (CRASH, "crash"),
])
def test_internal_failures_are_reported(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is True, verdict
    assert verdict["kind"] == kind
    assert verdict["signature"]


@pytest.mark.parametrize("output", [OOM, BAD_ALLOC])
def test_resource_exhaustion_is_not_a_bug(output):
    """The ordering guarantee.

    LLVM reports allocation failure through the same `LLVM ERROR:` channel
    as a genuine fatal error, so matching that channel without subtracting
    these first would file every out-of-memory as a compiler bug.
    """
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "resource"


def test_ordinary_diagnostic_is_not_a_bug():
    """The common case: fusing two modules produces ill-typed IR, and the
    verifier saying so is the tool working."""
    verdict = analyzer.classify(DIAGNOSTIC)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "diagnostic"


def test_clean_output_is_not_a_bug():
    assert analyzer.classify("")["is_bug"] is False
    assert analyzer.classify("module {\n}\n")["is_bug"] is False


def test_assert_signature_names_the_source_location():
    sig = analyzer.crash_signature(ASSERT)
    assert "Dialect.cpp:1234" in sig
    assert "/src/" not in sig     # build-directory prefix is not stable


def test_distinct_assertions_get_distinct_signatures():
    other = ASSERT.replace("Dialect.cpp:1234", "Coalesce.cpp:77")
    assert analyzer.crash_signature(ASSERT) != analyzer.crash_signature(other)


def test_crash_signature_skips_the_reporting_frames():
    """The top frames are LLVM's own signal handler and identical for every
    crash; the first frame below them is what distinguishes two bugs."""
    sig = analyzer.crash_signature(CRASH)
    assert "PrintStackTrace" not in sig
    assert "TritonGPUCoalescePass" in sig


# ---------------------------------------------------------------------------
# RUN-line parsing — the pipeline is what the test exercises
# ---------------------------------------------------------------------------

def test_run_line_yields_the_pass_pipeline():
    src = ("// RUN: triton-opt %s -split-input-file -tritongpu-coalesce "
           "| FileCheck %s\nmodule {}\n")
    facts = analyzer.analyze_seed(src)
    assert facts["passes"] == ["-tritongpu-coalesce"]
    assert facts["needs_split"] is True
    assert facts["has_run_line"] is True


def test_filecheck_arguments_are_not_passed_to_triton_opt():
    """Everything after the first pipe belongs to FileCheck."""
    src = ("// RUN: triton-opt %s -canonicalize | FileCheck %s "
           "--check-prefix=CHECK-DAG --dump-input-context=20\nmodule {}\n")
    assert analyzer.analyze_seed(src)["passes"] == ["-canonicalize"]


def test_pass_options_are_kept():
    """`-tritongpu-pipeline=num-stages=3` is a different test from
    `-tritongpu-pipeline`."""
    src = "// RUN: triton-opt %s -tritongpu-pipeline=num-stages=3\nmodule {}\n"
    assert analyzer.analyze_seed(src)["passes"] == ["-tritongpu-pipeline=num-stages=3"]


def test_seeds_without_a_pipeline_are_still_valid():
    """`triton-opt %s | FileCheck %s` is a parse-and-print round trip, which
    is a real test of the IR printer."""
    src = "// RUN: triton-opt --split-input-file %s | FileCheck %s\nmodule {}\n"
    facts = analyzer.analyze_seed(src)
    assert facts["has_run_line"] is True
    assert facts["passes"] == []


def test_aliases_and_module_attrs_are_recorded():
    src = ('// RUN: triton-opt %s -canonicalize\n'
           '#blocked = #ttg.blocked<{sizePerThread = [1]}>\n'
           'module attributes {"ttg.num-warps" = 4 : i32} {\n'
           '  tt.func @kernel() { tt.return }\n}\n')
    facts = analyzer.analyze_seed(src)
    assert facts["aliases"] == ["blocked"]
    assert "ttg.num-warps" in facts["module_attrs"]
    assert facts["func_names"] == ["kernel"]


# ---------------------------------------------------------------------------
# Section splitting — what makes the corpus valid at all
# ---------------------------------------------------------------------------

SECTIONED = """\
// RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1]}>
module { tt.func @a() { tt.return } }

// -----

#blocked = #ttg.blocked<{sizePerThread = [2]}>
module { tt.func @b() { tt.return } }
"""


def test_sections_are_split_into_separate_seeds():
    """A `-split-input-file` test is several independent modules that lit
    runs one at a time, and they routinely reuse names across sections.

    Kept whole the file is not valid IR — the second `#blocked` is a
    redefinition and the module fails to parse before any pass runs.
    Measured on the real corpus: 129 of 247 usable tests define an alias
    more than once, every one of them a sectioned test, and 154 of 200
    fused pairs collided.
    """
    sections = setup_mod._split_sections(SECTIONED)
    assert len(sections) == 2
    alias_re = re.compile(r"^#(\w+)\s*=", re.M)
    for s in sections:
        names = alias_re.findall(s)
        assert len(names) == len(set(names)), s


def test_each_section_keeps_the_run_line():
    """The pipeline describes every section, not just the first."""
    for s in setup_mod._split_sections(SECTIONED):
        assert "RUN: triton-opt" in s


def test_unsectioned_input_is_returned_unchanged():
    src = "// RUN: triton-opt %s -canonicalize\nmodule {}\n"
    assert setup_mod._split_sections(src) == [src]


# ---------------------------------------------------------------------------
# Fusion wiring
# ---------------------------------------------------------------------------

def test_triton_reuses_the_mlir_strategies():
    """triton-opt reads and writes the same textual MLIR, so it reuses the
    MLIR strategies verbatim — the arrangement GCC has with clang's."""
    for kw in ("dataflow_fusion", "state_fusion", "declaration_fusion"):
        assert get_strategies("triton", **{kw: True}), kw


def test_parser_emits_dataflow_metadata():
    """Without variables/dataflows the dataflow strategy is a silent no-op,
    which is what projects/mlir/parser.py does today."""
    parser = _load("ffl_triton_parser_test", "projects/triton/parser.py")
    src = ('module { tt.func @k(%arg0: i32) {\n'
           '  %0 = arith.addi %arg0, %arg0 : i32\n'
           '  %1 = arith.muli %0, %arg0 : i32\n'
           '  tt.return } }\n')
    meta = parser._parser.parse_content(src, "x.mlir")
    assert meta["variables"], meta
    assert meta["dataflows"], meta


def test_driver_withholds_oracle_breaking_flags():
    """-verify-diagnostics makes triton-opt exit non-zero unless the errors
    marked in the source all fire; on a fused module those markers no
    longer correspond to anything, so every run would "fail"."""
    driver_mod = _load("ffl_triton_driver_test", "projects/triton/driver.py")
    drv = driver_mod.TritonDriver({"execution": {}})
    chosen = drv._choose_passes({"passes": ["-canonicalize", "-verify-diagnostics"]})
    assert "-verify-diagnostics" not in chosen


def test_driver_always_allows_unregistered_dialects():
    """53 of Triton's own tests need it; without it their IR does not parse
    and the run tests nothing. It is harmless otherwise."""
    driver_mod = _load("ffl_triton_driver_test2", "projects/triton/driver.py")
    drv = driver_mod.TritonDriver({"execution": {}})
    cmd = drv._build_command("/tmp/x.mlir", {"passes": ["-canonicalize"]})
    assert "-allow-unregistered-dialect" in cmd
