"""
Tests for the tint adapter: the crash oracle and the WGSL wiring.

tint is a *compiler*, not an executor, so the shape of the problem differs
from the JS adapters: the expected outcome of a fused (and usually
ill-formed) shader is a clean diagnostic, not an exception. The oracle's
job is separating tint's own internal errors from that flood.

Both internal shapes below are taken from src/tint/utils/ice/ice.cc:
`InternalCompilerError::Error()` returns
`File() + ":" + Line() + " internal compiler error: " + Message()`.
"""

import importlib.util
import os
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


analyzer = _load("ffl_tint_analyzer_test", "projects/tint/analyzer.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

ICE = ("../../src/tint/lang/core/ir/validator.cc:812 internal compiler error: "
       "TINT_ASSERT(value != nullptr)\n")
UNREACHABLE = ("../../src/tint/lang/msl/writer/printer.cc:99 internal compiler "
               "error: TINT_UNREACHABLE unhandled type\n")
UNIMPLEMENTED = ("../../src/tint/lang/hlsl/writer/printer.cc:42 internal compiler "
                 "error: TINT_UNIMPLEMENTED subgroup matrix\n")
ASAN = """==1234==ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0
SUMMARY: AddressSanitizer: heap-use-after-free src/tint/lang/core/ir/module.cc:88 in tint::core::ir::Module::Destroy
"""
UBSAN = "../../src/tint/utils/math/hash.h:45:12: runtime error: left shift of negative value -1"

OOM = "terminate called after throwing an instance of 'std::bad_alloc'\n  what():  std::bad_alloc\n"
STACK = "Stack overflow\n"

DIAGNOSTIC = "shader.wgsl:3:5 error: unresolved identifier 'foo'\n"
DIAGNOSTIC2 = ("shader.wgsl:12:9 error: no matching call to 'textureSample(f32)'\n"
               "shader.wgsl:1:1 error: must be a struct\n")


@pytest.mark.parametrize("output,kind", [
    (ICE, "ice"),
    (UNREACHABLE, "ice"),
    (ASAN, "sanitizer"),
    (UBSAN, "ubsan"),
])
def test_real_failures_are_reported(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is True, verdict
    assert verdict["kind"] == kind
    assert verdict["signature"]


@pytest.mark.parametrize("output", [DIAGNOSTIC, DIAGNOSTIC2])
def test_ordinary_diagnostics_are_not_bugs(output):
    """A fused WGSL program is usually ill-formed; rejecting it cleanly is
    tint working, not tint failing."""
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "diagnostic"


@pytest.mark.parametrize("output", [OOM, STACK])
def test_resource_exhaustion_is_not_a_bug(output):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "resource"


def test_clean_output_is_not_a_bug():
    assert analyzer.classify("")["is_bug"] is False
    assert analyzer.classify("; SPIR-V\n; Version: 1.3\n")["is_bug"] is False


def test_ice_signature_names_the_source_location():
    """The build embeds paths relative to its own out directory; only the
    src/tint-relative tail is stable across machines."""
    sig = analyzer.crash_signature(ICE)
    assert sig == "ICE: src/tint/lang/core/ir/validator.cc:812"
    assert ".." not in sig


def test_ice_kind_distinguishes_the_macro():
    assert analyzer.crash_signature(UNREACHABLE).startswith("UNREACHABLE:")
    assert analyzer.crash_signature(ICE).startswith("ICE:")


def test_unimplemented_is_not_a_bug():
    """TINT_UNIMPLEMENTED *declares* that a path was never written —
    `default: TINT_IR_UNIMPLEMENTED(mod) << builtin.value()` and its kin.
    Reaching one means the input used a feature tint does not support yet,
    which is documented behaviour, not a defect; filing them would add one
    entry per unsupported builtin.

    TINT_ASSERT and TINT_UNREACHABLE are the opposite: they assert
    something the compiler believes cannot happen, so reaching one is
    always a real internal error. The campaign's first run produced a
    `TINT_UNIMPLEMENTED subgroup_id` finding, which is what prompted the
    distinction."""
    verdict = analyzer.classify(UNIMPLEMENTED)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "unimplemented"
    assert verdict["signature"] is None


def test_signature_is_stable_across_volatile_detail():
    a = analyzer.crash_signature(ICE)
    b = analyzer.crash_signature(
        "/other/build/" + ICE.replace("TINT_ASSERT(value != nullptr)",
                                      "TINT_ASSERT(value != nullptr) 0xdeadbeef"))
    assert a == b and a is not None


def test_distinct_ices_get_distinct_signatures():
    other = ICE.replace("validator.cc:812", "builder.cc:77")
    assert analyzer.crash_signature(ICE) != analyzer.crash_signature(other)


def test_ice_wins_over_an_ordinary_diagnostic():
    """tint prints diagnostics before it crashes; the ICE must not be lost
    behind them."""
    verdict = analyzer.classify(DIAGNOSTIC + ICE)
    assert verdict["kind"] == "ice"
    assert verdict["is_bug"] is True


def test_resource_exhaustion_wins_over_a_signal():
    """A bad_alloc that then aborts is not a compiler bug."""
    verdict = analyzer.classify(OOM + "Aborted (core dumped)\n")
    assert verdict["is_bug"] is False
    assert verdict["kind"] == "resource"


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

def test_entry_points_are_detected():
    """A backend only compiles an entry point; a shader with none exercises
    the front end alone."""
    facts = analyzer.analyze_seed(
        "@fragment fn f() {}\n@compute @workgroup_size(1) fn c() {}\n")
    assert facts["entry_stages"] == ["compute", "fragment"]
    assert facts["has_entry"] is True
    assert analyzer.analyze_seed("fn helper() {}\n")["has_entry"] is False


def test_enable_directives_are_collected():
    facts = analyzer.analyze_seed("enable f16;\nrequires readonly_and_readwrite_storage_textures;\n")
    assert "f16" in facts["enables"]


def test_overrides_are_detected():
    assert analyzer.analyze_seed("override w: u32 = 1;\n")["uses_overrides"] is True
    assert analyzer.analyze_seed("const w: u32 = 1;\n")["uses_overrides"] is False


# ---------------------------------------------------------------------------
# Fusion wiring
# ---------------------------------------------------------------------------

def test_tint_reuses_the_wgsl_strategies():
    """tint and naga are the same language, so they share the three
    source-to-source strategies — the arrangement GCC has with clang's.
    Only the driver and the crash oracle differ."""
    for kw in ("dataflow_fusion", "state_fusion", "declaration_fusion"):
        strategies = get_strategies("tint", **{kw: True})
        assert strategies, kw
        assert type(strategies[0]).__name__.startswith("Naga")


def test_fusion_produces_wgsl_children():
    strategy = get_strategies("tint", dataflow_fusion=True)[0]
    child = strategy.fuse(
        Seed(content="fn helper(x: f32) -> f32 { return x * 2.0; }\n"
                     "@fragment fn main() { let y = helper(1.0); }\n"),
        Seed(content="struct S { a: f32 }\nfn other(s: S) -> f32 { return s.a; }\n"))
    assert child and child.content.strip()
    assert (child.metadata or {}).get("extension") == ".wgsl"


def test_parser_emits_dataflow_metadata():
    """A strategy that finds these missing degrades to a silent no-op."""
    import projects.tint.parser as parser
    meta = parser._parser.parse_content(
        "fn helper(x: f32) -> f32 {\n  let y = x * 2.0;\n  return y;\n}\n")
    assert meta["type"] == "wgsl"
    assert meta["variables"] and meta["dataflows"]
    assert "helper" in meta["functions"]


def test_driver_does_not_use_ulimit_v():
    """`ulimit -v` and ASan are incompatible: ASan reserves ~20 TB of
    virtual address space, so a useful -v cap kills tint before it runs."""
    driver_mod = _load("ffl_tint_driver_test", "projects/tint/driver.py")
    drv = driver_mod.TintDriver({"execution": {"mem_limit_mb": 512}})
    cmd = drv._build_command("/tmp/x.wgsl", {"entry_stages": ["fragment"]})
    assert "ulimit -v" not in cmd
    assert "hard_rss_limit_mb=512" in cmd
    assert "--format" in cmd


def test_driver_reaches_every_backend():
    """A translation bug lives in one code writer, so all of them must be
    reachable — for a shader that has an entry point to compile."""
    import random
    driver_mod = _load("ffl_tint_driver_backends", "projects/tint/driver.py")
    drv = driver_mod.TintDriver({"execution": {}})
    random.seed(7)
    seen = set()
    for _ in range(400):
        backend, _args = drv._choose_args({"entry_stages": ["fragment"]})
        seen.add(backend)
    assert seen == set(drv.BACKENDS), seen


def test_shader_without_entry_point_goes_to_the_wgsl_writer():
    """No entry point means no code writer has anything to emit, so those
    runs would only exercise the front end. The WGSL writer round-trips the
    whole module through the IR and still does real work."""
    import random
    driver_mod = _load("ffl_tint_driver_noentry", "projects/tint/driver.py")
    drv = driver_mod.TintDriver({"execution": {}})
    random.seed(11)
    for _ in range(50):
        backend, args = drv._choose_args({"entry_stages": []})
        assert backend == "wgsl"
        assert args[0] == "--format wgsl"


# ---------------------------------------------------------------------------
# Ordering: the ICE, not ASan's report of the trap it performs
# ---------------------------------------------------------------------------

# TINT_ICE ends in __builtin_trap(). Under an ASan build ASan intercepts
# that trap and prints its own report naming ice.cc — identical for every
# internal error in the compiler, whatever actually failed.
ICE_UNDER_ASAN = (
    "/b/dawn/src/tint/lang/wgsl/writer/ir_to_program/ir_to_program.cc:1110 "
    "internal compiler error: Switch() matched no cases\n"
    "==4658==ERROR: AddressSanitizer: ILL on unknown address 0x559b271fbc09\n"
    "SUMMARY: AddressSanitizer: ILL /b/dawn/src/tint/utils/ice/ice.cc:71 "
    "in tint::InternalCompilerError::~InternalCompilerError()\n")


def test_ice_wins_over_asans_report_of_its_own_trap():
    """Otherwise every distinct ICE collapses into one group.

    Taking ASan's summary would give `ice.cc:71 in ~InternalCompilerError`
    as the signature for every internal error tint can raise, hiding where
    the failure actually was and defeating deduplication entirely.
    """
    verdict = analyzer.classify(ICE_UNDER_ASAN)
    assert verdict["kind"] == "ice"
    assert verdict["signature"] == (
        "ICE: src/tint/lang/wgsl/writer/ir_to_program/ir_to_program.cc:1110")
    assert "ice.cc" not in verdict["signature"]


def test_genuine_memory_errors_still_reported():
    """Ordering the ICE first must not swallow a real memory bug, which
    carries no `internal compiler error` line above it."""
    verdict = analyzer.classify(ASAN)
    assert verdict["kind"] == "sanitizer"
    assert "heap-use-after-free" in verdict["signature"]


def test_two_distinct_ices_do_not_collapse_under_asan():
    other = ICE_UNDER_ASAN.replace("ir_to_program.cc:1110", "builder.cc:42")
    assert (analyzer.crash_signature(ICE_UNDER_ASAN)
            != analyzer.crash_signature(other))


def test_bundle_uses_the_wgsl_extension():
    """A saved reproducer must carry the extension its compiler needs.

    _seed_extension names a handful of projects explicitly and fell back to
    ".txt" for everything else, so every tint bundle was written as
    test.txt while its own test.sh named a .wgsl path — the reproducer
    could not be re-run without renaming it by hand. The parser already
    records the right extension; the fallback now uses it.
    """
    from core.orchestrator import FusionFuzzLoop

    class _Loop:
        project_name = "tint"

    class _Seed:
        metadata = {"extension": ".wgsl"}

    assert FusionFuzzLoop._seed_extension(_Loop(), _Seed()) == ".wgsl"


def test_unknown_project_still_falls_back_to_txt():
    """The fallback only fires when the parser recorded an extension."""
    from core.orchestrator import FusionFuzzLoop

    class _Loop:
        project_name = "some-new-target"

    class _Seed:
        metadata = {}

    assert FusionFuzzLoop._seed_extension(_Loop(), _Seed()) == ".txt"


# ---------------------------------------------------------------------------
# Sanitizer signatures must name tint's code, not libstdc++
# ---------------------------------------------------------------------------

SEGV_THROUGH_STDLIB = (
    "/b/dawn/src/tint/lang/wgsl/writer/ir_to_program/ir_to_program.cc:369:64: "
    "runtime error: member call on null pointer of type 'struct StatementList'\n"
    "AddressSanitizer:DEADLYSIGNAL\n"
    "==1==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000058\n"
    "    #0 0x55 in std::__atomic_base<unsigned int>::load(std::memory_order) const "
    "/usr/include/c++/13/bits/atomic_base.h:505\n"
    "    #2 0x55 in tint::Vector<x>::Push(y) /b/dawn/src/tint/utils/containers/vector.h:697\n"
    "SUMMARY: AddressSanitizer: SEGV /usr/include/c++/13/bits/atomic_base.h:505 "
    "in std::__atomic_base<unsigned int>::load(std::memory_order) const\n")


def test_sanitizer_signature_names_tint_not_the_standard_library():
    """ASan's SUMMARY names wherever the faulting instruction happened to
    be. For a null dereference through a container that is a libstdc++
    header, so the signature became
    `atomic_base.h:505 in std::__atomic_base<unsigned int>::load` for a bug
    whose real location is ir_to_program.cc:369 — grouping by standard
    library internals rather than by the defect. UBSan's own line, or the
    first tint frame in the backtrace, is used instead.
    """
    sig = analyzer.crash_signature(SEGV_THROUGH_STDLIB)
    assert "atomic_base" not in sig, sig
    assert "src/tint/lang/wgsl/writer/ir_to_program/ir_to_program.cc:369" in sig


def test_null_deref_gets_one_signature_whichever_sanitizer_reports_it():
    """A null dereference through a container is always reported by UBSan
    and *sometimes* also by ASan as a SEGV, depending on whether ASan gets
    to print before the process dies.

    Keying the two cases differently filed one bug under two names — a
    16-hour run produced both `AddressSanitizer__SEGV_at_...:369` and
    `UBSAN__member_call_on_null_pointer_of_type__struct_StatementList_`
    for the same fault at the same line.
    """
    ubsan_only = SEGV_THROUGH_STDLIB.split("AddressSanitizer:DEADLYSIGNAL")[0]
    assert (analyzer.crash_signature(ubsan_only)
            == analyzer.crash_signature(SEGV_THROUGH_STDLIB))
    assert "ir_to_program.cc:369" in analyzer.crash_signature(ubsan_only)


def test_a_real_use_after_free_keeps_its_own_kind():
    """Unifying the null-deref case must not blur distinct memory bugs."""
    sig = analyzer.crash_signature(ASAN)
    assert "heap-use-after-free" in sig
    assert "NULL-DEREF" not in sig


def test_sanitizer_signature_keeps_the_fault_kind():
    """The kind comes from ASan's ERROR line, not from SUMMARY's first
    token, which is a path when SUMMARY already points into tint."""
    sig = analyzer.crash_signature(ASAN)
    assert "heap-use-after-free" in sig, sig


def test_distinct_memory_bugs_do_not_collide():
    other = ASAN.replace("src/tint/lang/core/ir/module.cc:88",
                         "src/tint/lang/hlsl/writer/printer/printer.cc:570")
    assert analyzer.crash_signature(ASAN) != analyzer.crash_signature(other)
