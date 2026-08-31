"""
Tests for the SpiderMonkey adapter: the crash oracle, the jit-test
directives, and the execution harness.

SpiderMonkey is an *executing* target, so most of what a fused program
prints is an ordinary JavaScript exception — the expected result of
joining two unrelated scripts. The oracle's job is separating the few
outputs that are engine bugs from that flood.

What makes this engine different from V8 is that its corpus *declares*
which failures are correct: a jit-test file can say `allow-oom` or
`error: TypeError`, and honouring that is what keeps a deliberate failure
from being filed as a finding.
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


analyzer = _load("ffl_sm_analyzer_test", "projects/spidermonkey/analyzer.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

# Both shapes come from mfbt/Assertions.h:
#   fprintf(stderr, "[%d] Assertion failure: %s, at %s:%d\n", ...)
#   fprintf(stderr, "[%d] Hit MOZ_CRASH(%s) at %s:%d\n", ...)
MOZ_ASSERT = ("[31337] Assertion failure: obj->is<NativeObject>(), "
              "at /builds/worker/checkouts/gecko/js/src/jit/Lowering.cpp:1234\n")
MOZ_CRASH = ("[31337] Hit MOZ_CRASH(unreachable code) at "
             "/builds/worker/checkouts/gecko/js/src/jit/Ion.cpp:99\n")
ASAN = """==31337==ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0
SUMMARY: AddressSanitizer: heap-use-after-free js/src/gc/Marking.cpp:88 in js::GCMarker::traverse
"""
UBSAN = "js/src/jit/MacroAssembler.cpp:45:12: runtime error: left shift of negative value -1"

OOM = "out of memory\n"
OOM_CRASH = ("[31337] Hit MOZ_CRASH(out of memory) at "
             "/builds/worker/checkouts/gecko/js/src/gc/Nursery.cpp:12\n")
RECURSION = "InternalError: too much recursion\n"

TYPE_ERROR = "/tmp/ffl/test.js:12:9 TypeError: obj.foo is not a function\n"
REF_ERROR = "uncaught exception: ReferenceError: quux is not defined\n"


@pytest.mark.parametrize("output,kind", [
    (MOZ_ASSERT, "assert"),
    (MOZ_CRASH, "moz_crash"),
    (ASAN, "sanitizer"),
    (UBSAN, "ubsan"),
])
def test_real_failures_are_reported(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is True
    assert verdict["kind"] == kind
    assert verdict["signature"]


@pytest.mark.parametrize("output", [OOM, OOM_CRASH, RECURSION])
def test_resource_exhaustion_is_not_a_bug(output):
    """The ordering guarantee.

    SpiderMonkey reports some out-of-memory conditions through MOZ_CRASH,
    the same channel a genuine impossible-state abort uses, so the
    resource shapes have to be subtracted *before* the crash handling.
    """
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "resource"


@pytest.mark.parametrize("output", [TYPE_ERROR, REF_ERROR])
def test_ordinary_js_exceptions_are_not_bugs(output):
    assert analyzer.classify(output)["is_bug"] is False


def test_clean_output_is_not_a_bug():
    assert analyzer.classify("")["is_bug"] is False
    assert analyzer.classify("done\n")["is_bug"] is False


def test_assertion_signature_names_the_source_location():
    sig = analyzer.crash_signature(MOZ_ASSERT)
    # The build directory differs between machines; only the js/src-relative
    # tail is a stable grouping key.
    assert sig == "ASSERT: js/src/jit/Lowering.cpp:1234"
    assert "builds/worker" not in sig


def test_distinct_assertions_get_distinct_signatures():
    other = MOZ_ASSERT.replace("Lowering.cpp:1234", "CodeGenerator.cpp:77")
    assert analyzer.crash_signature(MOZ_ASSERT) != analyzer.crash_signature(other)


def test_seed_declared_unhandlable_oom_is_honoured():
    """A test that declares it may abort on OOM is describing that crash
    as its correct behaviour — the same class of false positive as V8's
    triggerAssertFalse."""
    facts = analyzer.analyze_seed(
        "// |jit-test| allow-oom; allow-unhandlable-oom\ngc();\n")
    assert facts["allow_unhandlable_oom"] is True
    crash = ("[31337] Hit MOZ_CRASH(Failed to allocate memory) at "
             "/b/gecko/js/src/gc/Heap.cpp:5\n")
    assert analyzer.classify(crash, facts)["is_bug"] is False
    # Without the declaration the same output is a finding.
    assert analyzer.classify(crash, {})["is_bug"] is True


# ---------------------------------------------------------------------------
# jit-test directives
# ---------------------------------------------------------------------------

def test_directives_are_parsed_into_flags_and_expectations():
    facts = analyzer.analyze_seed(
        "// |jit-test| --ion-eager; --no-threads; allow-oom; error: TypeError\n")
    assert facts["flags"] == ["--ion-eager", "--no-threads"]
    assert facts["allow_oom"] is True
    assert facts["expects_error"] == "TypeError"


def test_path_bearing_and_absurd_flags_are_rejected():
    """A flag naming a file would point at the test's original directory;
    an absurd numeric value aborts the shell during startup, which
    reproduces on an empty script and says nothing about the engine."""
    facts = analyzer.analyze_seed(
        "// |jit-test| --ion-eager; --thread-count=99999999999\n")
    assert facts["flags"] == ["--ion-eager"]


def test_module_directive_is_detected():
    assert analyzer.analyze_seed("// |jit-test| module\n")["is_module"] is True


# ---------------------------------------------------------------------------
# Fusion wiring and the harness
# ---------------------------------------------------------------------------

def test_spidermonkey_reuses_the_javascript_strategies():
    """Same language as V8, so the same source-to-source strategies — the
    arrangement GCC has with clang's. Only the driver and oracle differ."""
    for kw in ("dataflow_fusion", "state_fusion", "declaration_fusion"):
        strategies = get_strategies("spidermonkey", **{kw: True})
        assert strategies, kw


def test_fusion_output_carries_no_harness():
    """The harness is applied by the driver at execution time.

    Baking it into the child would compound across a fusion chain and
    offer its own identifiers to the dataflow rename — a defect caught in
    the V8 adapter, where a seed's variable was renamed to the harness's
    `ARGS`.
    """
    for kw in ("dataflow_fusion", "state_fusion"):
        strategy = get_strategies("spidermonkey", **{kw: True})[0]
        child = strategy.fuse(
            Seed(content="function u(x){ return x; }\nu(1);\n"),
            Seed(content="function v(y){ return y; }\nv(2);\n"))
        for marker in ("__ffl_exercise", "__FFL_NAMES"):
            assert marker not in child.content, (kw, marker)


def test_harness_keeps_the_directive_line_first():
    driver_mod = _load("ffl_sm_driver_test", "projects/spidermonkey/driver.py")
    wrapped = driver_mod.apply_harness(
        "// |jit-test| --ion-eager\nfunction a(x) { return x; }\n")
    assert wrapped.splitlines()[0] == "// |jit-test| --ion-eager"
    assert '__FFL_NAMES = ["a"]' in wrapped
    assert "function a(x) { return x; }" in wrapped


def test_harness_collects_hoisted_and_bound_declarations():
    """Lexically, not by diffing globalThis: `function f(){}` is hoisted,
    so a before/after snapshot would classify it as a built-in and skip
    exactly the functions worth calling."""
    driver_mod = _load("ffl_sm_driver_names", "projects/spidermonkey/driver.py")
    names = driver_mod._toplevel_callables(
        "function f(){}\n"
        "var g = function(){};\n"
        "let h = (x) => x;\n"
        "const k = x => x;\n"
        "class C {}\n"
        "  function indented(){}\n"
        "let plain = 42;\n")
    assert names == ["f", "g", "h", "k", "C"]


def test_harness_is_skipped_when_there_is_nothing_to_exercise():
    driver_mod = _load("ffl_sm_driver_skip", "projects/spidermonkey/driver.py")
    src = "let a = 1;\nprint(a);\n"
    assert driver_mod.apply_harness(src) == src


def test_memory_cap_does_not_use_ulimit_v():
    """`ulimit -v` and ASan are incompatible: ASan reserves ~20 TB of
    virtual address space, so a useful -v cap kills the shell before
    main() and every execution fails identically."""
    driver_mod = _load("ffl_sm_driver_mem", "projects/spidermonkey/driver.py")
    drv = driver_mod.SpiderMonkeyDriver({"execution": {"mem_limit_mb": 512}})
    cmd = drv._build_command("/tmp/x.js", {"flags": []})
    assert "ulimit -v" not in cmd
    assert "hard_rss_limit_mb=512" in cmd
    assert "--fuzzing-safe" in cmd


def test_seed_flags_reach_the_command():
    driver_mod = _load("ffl_sm_driver_flags", "projects/spidermonkey/driver.py")
    drv = driver_mod.SpiderMonkeyDriver({"execution": {}})
    chosen = drv._choose_flags({"flags": ["--ion-eager", "--enable-oom-breakpoint"]})
    assert "--ion-eager" in chosen
    # Turning on artificial OOM manufactures allocation failures the
    # oracle would then have to un-attribute.
    assert "--enable-oom-breakpoint" not in chosen


# ---------------------------------------------------------------------------
# Ordering: the assertion, not ASan's report of it
# ---------------------------------------------------------------------------

# Under an ASan build the deliberate abort MOZ_ASSERT performs is itself
# intercepted, so a real assertion failure arrives with an ASan report
# stapled to it. ASan's SUMMARY names MOZ_CrashSequence in Assertions.h —
# identical for every assertion in the engine.
ASSERT_UNDER_ASAN = (
    "[9201] Assertion failure: obj->is<NativeObject>(), at "
    "/b/gecko/js/src/jit/Lowering.cpp:1234\n"
    "==9201==ERROR: AddressSanitizer: SEGV on unknown address 0x000000000000\n"
    "SUMMARY: AddressSanitizer: SEGV "
    "/b/gecko/js/src/obj/dist/include/mozilla/Assertions.h:261:3 "
    "in MOZ_CrashSequence(void*, long)\n")


def test_assertion_wins_over_asans_report_of_it():
    """Otherwise every distinct assertion collapses into one group.

    Taking ASan's summary would give `Assertions.h:261 in MOZ_CrashSequence`
    as the signature for every assertion in the engine, hiding where the
    failure actually was and defeating deduplication entirely.
    """
    verdict = analyzer.classify(ASSERT_UNDER_ASAN)
    assert verdict["kind"] == "assert"
    assert verdict["signature"] == "ASSERT: js/src/jit/Lowering.cpp:1234"
    assert "MOZ_CrashSequence" not in verdict["signature"]


def test_genuine_sanitizer_reports_still_reported():
    """Ordering assert-first must not swallow a real memory error, which
    carries no assertion line."""
    verdict = analyzer.classify(ASAN)
    assert verdict["kind"] == "sanitizer"
    assert "heap-use-after-free" in verdict["signature"]


def test_bad_command_line_is_not_an_engine_bug():
    """A MOZ_CRASH while the shell parses its own arguments means the
    command line was wrong — it reproduces on an empty script.

    This one was real: --nursery-strings is declared with
    addStringOption("on/off"), and passing it bare aborted the shell
    before any JavaScript ran. Verifying that a flag exists is not enough;
    its arity has to be verified too.
    """
    out = ("[9201] Hit MOZ_CRASH(invalid option value for --nursery-strings, "
           "must be on/off) at shell/js.cpp:14540\n"
           "==9201==ERROR: AddressSanitizer: SEGV on unknown address 0x0\n")
    verdict = analyzer.classify(out)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "shell_option"


def test_value_taking_flags_carry_their_value():
    """A value-taking flag passed bare aborts the shell during argument
    parsing, so every execution that draws it is wasted."""
    driver_mod = _load("ffl_sm_driver_arity", "projects/spidermonkey/driver.py")
    for flag in (driver_mod.SpiderMonkeyDriver.GC_FLAGS
                 + driver_mod.SpiderMonkeyDriver.VERIFY_FLAGS
                 + driver_mod.SpiderMonkeyDriver.TIER_FLAGS
                 + driver_mod.SpiderMonkeyDriver.TIER_DOWN_FLAGS):
        for token in flag.split():
            name = token.split("=", 1)[0]
            if name in ("--gc-zeal", "--nursery-strings", "--spectre-mitigations",
                        "--ion-offthread-compile"):
                assert "=" in token, token


def test_harness_arms_the_shell_watchdog_first():
    """The harness must bound its own execution, and over the whole
    program rather than just the epilogue.

    Two things make a harnessed program run long: the epilogue calls
    functions with deliberately huge arguments (2**31 is the int32
    boundary, where the JIT's overflow paths are interesting), and the
    guard lets execution continue past an exception into loops that would
    never otherwise be reached. Arming only around the epilogue left the
    second uncovered — measured 3/60 timeouts against a 1/60 baseline,
    versus 1/60 once the watchdog covered everything.
    """
    driver_mod = _load("ffl_sm_driver_watchdog", "projects/spidermonkey/driver.py")
    wrapped = driver_mod.apply_harness("function f(n){ for(let i=0;i<n;i++); }\n")
    lines = [l for l in wrapped.splitlines() if l.strip()]
    assert "timeout(8)" in lines[0], lines[0]
    # Before the guarded body, not after it. (Compared against the body
    # text, not against "try {" — the watchdog line is itself a try/catch.)
    assert wrapped.index("timeout(8)") < wrapped.index("function f(n)")
    assert wrapped.index("timeout(8)") < wrapped.index("__ffl_exercise")


def test_watchdog_termination_is_not_a_bug():
    """The watchdog firing means the program ran long, not that the engine
    is broken."""
    verdict = analyzer.classify("Script terminated by interrupt handler.\n")
    assert verdict["is_bug"] is False
    assert verdict["kind"] == "resource"
