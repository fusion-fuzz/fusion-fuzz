"""
Tests for the V8 adapter: the crash oracle and the JavaScript structure
rules fusion has to respect.

The oracle carries most of the weight here. V8 is an *executing* target,
so the overwhelming majority of what a fused program prints is an ordinary
JavaScript exception — the expected result of joining two unrelated
scripts. A test suite that only checked "does it spot a DCHECK" would miss
the failure mode that actually matters: filing that flood as findings.

The other half is that V8 reports running out of memory through the very
same `# Fatal error` channel it uses for a genuine assertion failure, so
the two are separable only by order.
"""

import importlib.util
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.fusion import Seed, get_strategies, js_toplevel_names, split_js_file  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("ffl_v8_analyzer_test", "projects/v8/analyzer.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

# The shape V8_Fatal prints (src/base/logging.cc), with the
# "Debug check failed: %s." body a DCHECK produces.
DCHECK = """

#
# Fatal error in ../../src/compiler/backend/instruction-selector.cc, line 1234
# Debug check failed: !instr->HasOutput() || instr->Output()->IsRegister().
#
#FailureMessage Object: 0x7ffd0a1b2c30
==== C stack trace ===============================

    v8::base::debug::StackTrace::StackTrace()
    v8::base::(anonymous namespace)::DefaultDcheckHandler()
    v8::internal::compiler::InstructionSelector::VisitNode()
    v8::internal::compiler::PipelineImpl::Run()
Received signal 4 ILL_ILLOPN
"""

SANDBOX = """

## V8 sandbox violation detected!

Faulting address: 0x000042000000
    v8::internal::JSObject::SetProperty()
"""

ASAN = """==12345==ERROR: AddressSanitizer: heap-use-after-free on address 0x60300000eff0
READ of size 8 at 0x60300000eff0 thread T0
    #0 0x5591 in v8::internal::Heap::CollectGarbage()
SUMMARY: AddressSanitizer: heap-use-after-free src/heap/heap.cc:1234 in v8::internal::Heap::CollectGarbage
"""

UBSAN = "../../src/objects/smi.h:45:12: runtime error: left shift of negative value -1"

# The trap: V8 routes OOM through V8_Fatal, so this carries the exact
# header a real DCHECK failure has.
OOM = """

#
# Fatal javascript OOM in Reached heap limit
#
"""

OOM_ALLOC = """

#
# Fatal process out of memory: Reached heap limit
#
"""

STACK_OVERFLOW = "/tmp/x/test.js:3: RangeError: Maximum call stack size exceeded"

# Ordinary JavaScript exceptions: the *expected* outcome of fusing two
# unrelated programs, and the thing the oracle must not report.
TYPE_ERROR = """/tmp/ffl/test.js:12: TypeError: obj.foo is not a function
    obj.foo();
        ^
TypeError: obj.foo is not a function
    at /tmp/ffl/test.js:12:9
"""

REF_ERROR = """/tmp/ffl/test.js:4: ReferenceError: quux is not defined
    quux(1);
    ^
"""

SYNTAX_ERROR = """/tmp/ffl/test.js:7: SyntaxError: Identifier 'a' has already been declared
let a = 2;
^
"""


@pytest.mark.parametrize("output,kind", [
    (DCHECK, "dcheck"),
    (SANDBOX, "sandbox"),
    (ASAN, "sanitizer"),
    (UBSAN, "ubsan"),
])
def test_real_failures_are_reported(output, kind):
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is True
    assert verdict["kind"] == kind
    assert verdict["signature"]


@pytest.mark.parametrize("output", [OOM, OOM_ALLOC, STACK_OVERFLOW])
def test_resource_exhaustion_is_not_a_bug(output):
    """The ordering guarantee.

    OOM arrives with the same `# Fatal error` header as a DCHECK failure,
    so recognising the fatal shape first would file every out-of-memory as
    an assertion failure — the exact false-positive class that made six of
    seven CPython findings bogus.
    """
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "resource"
    assert verdict["signature"] is None


@pytest.mark.parametrize("output", [TYPE_ERROR, REF_ERROR, SYNTAX_ERROR])
def test_ordinary_js_exceptions_are_not_bugs(output):
    """The common case by a wide margin: fusion produced nonsense."""
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "exception"


def test_clean_output_is_not_a_bug():
    assert analyzer.classify("done\n")["is_bug"] is False
    assert analyzer.classify("")["is_bug"] is False


def test_signature_is_stable_across_volatile_detail():
    """Addresses and temp paths differ every run; the signature must not."""
    a = analyzer.crash_signature(DCHECK)
    b = analyzer.crash_signature(
        DCHECK.replace("0x7ffd0a1b2c30", "0x7f00deadbeef"))
    assert a == b and a is not None


def test_dcheck_signature_names_the_source_location():
    sig = analyzer.crash_signature(DCHECK)
    assert "instruction-selector.cc" in sig and "1234" in sig


def test_distinct_dchecks_get_distinct_signatures():
    other = DCHECK.replace("instruction-selector.cc, line 1234",
                           "js-call-reducer.cc, line 99")
    assert analyzer.crash_signature(DCHECK) != analyzer.crash_signature(other)


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

def test_natives_syntax_is_detected():
    """%-syntax is a *parse* error without --allow-natives-syntax, so
    missing this makes the seed unrunnable rather than merely unoptimised."""
    facts = analyzer.analyze_seed("%OptimizeFunctionOnNextCall(f);")
    assert facts["uses_natives"] is True
    assert facts["optimize_score"] >= 1


def test_test_flags_are_honoured():
    facts = analyzer.analyze_seed("// Flags: --allow-natives-syntax --expose-gc\nf();")
    assert "--allow-natives-syntax" in facts["flags"]
    assert "--expose-gc" in facts["flags"]


def test_path_bearing_flags_are_rejected():
    """A flag naming a file would point at the test's original directory,
    which the seed has left."""
    facts = analyzer.analyze_seed("// Flags: --snapshot-blob=/some/where.bin --expose-gc\n")
    assert facts["flags"] == ["--expose-gc"]


# ---------------------------------------------------------------------------
# The structural rule fusion has to respect
# ---------------------------------------------------------------------------

def test_toplevel_names_are_scoped_by_brace_depth():
    """Column is not scope.

    mjsunit routinely wraps a section in a bare block and leaves the body
    unindented. A `const` in there is block-scoped, so two of them in one
    file are legal — counting them as top-level would make the strategy
    rename bindings that never collided.
    """
    src = "const a = 1;\n{\nconst b = 2;\n}\n{\nconst b = 3;\n}\n"
    assert js_toplevel_names(src) == {"a"}


def test_toplevel_names_ignore_strings_and_comments():
    src = 'const real = 1;\n// const commented = 2;\nconst s = "const inside = 3;";\n'
    assert js_toplevel_names(src) == {"real", "s"}


def test_var_and_function_are_not_deduped():
    """Redeclaring them is legal, so renaming would be churn that changes
    the program without fixing anything."""
    assert js_toplevel_names("var x = 1;\nfunction f() {}\n") == set()


def test_lexical_redeclaration_is_removed_by_fusion():
    """A redeclared let/const is a SyntaxError thrown while parsing, so it
    takes down *both* halves before either runs."""
    strategy = get_strategies("v8", dataflow_fusion=True)[0]
    a = Seed(content="const shared = 1;\nconsole.log(shared);\n")
    b = Seed(content="const shared = 2;\nconsole.log(shared);\n")
    child = strategy.fuse(a, b)
    _, _, body = split_js_file(child.content)
    import re
    decls = re.findall(r"^const\s+([A-Za-z_$][\w$]*)", body, re.M)
    assert len(decls) == 2, body           # both halves kept
    assert len(set(decls)) == 2, body      # under distinct names
    # The donor's *use* has to move with its declaration, or the rename
    # trades a SyntaxError for a ReferenceError.
    renamed = [d for d in decls if d != "shared"][0]
    assert body.count(renamed) == 2, body


def test_flags_are_merged_not_dropped():
    """A dropped flag silently turns that half into a script that runs in
    the interpreter and tests nothing."""
    strategy = get_strategies("v8", dataflow_fusion=True)[0]
    a = Seed(content="// Flags: --allow-natives-syntax\nlet p = 1;\n")
    b = Seed(content="// Flags: --expose-gc\nlet q = 2;\n")
    child = strategy.fuse(a, b)
    header = child.content.splitlines()[0]
    assert header.startswith("// Flags:")
    assert "--allow-natives-syntax" in header and "--expose-gc" in header


def test_use_strict_is_hoisted():
    """A directive prologue only counts as one when it is the first
    statement; concatenation would bury the second file's and silently
    change that half's semantics."""
    strategy = get_strategies("v8", dataflow_fusion=True)[0]
    a = Seed(content='"use strict";\nlet p = 1;\n')
    b = Seed(content='"use strict";\nlet q = 2;\n')
    child = strategy.fuse(a, b)
    lines = [l for l in child.content.splitlines() if l.strip()]
    assert lines[0] == '"use strict";'
    assert sum(1 for l in lines if l.strip() == '"use strict";') == 1


def test_declaration_fusion_actually_injects_when_viable():
    """is_viable_pair must imply an injection.

    Without that the run reports declaration fusions that were really
    plain concatenation.
    """
    strategy = get_strategies("v8", declaration_fusion=True)[0]
    host = Seed(content="let o = new Object();\nlet t = o;\n")
    donor = Seed(content="class Donor {\n  constructor() { this.x = 1; }\n}\n")
    assert strategy.is_viable_pair(host, donor)
    child = strategy.fuse(host, donor)
    assert child.metadata.get("injected_class") == "Donor"
    assert "new Donor(" in child.content


def test_declaration_fusion_accepts_constructor_functions():
    """The pre-ES6 constructor is as common as `class` in this corpus and
    V8 treats objects from the two identically."""
    strategy = get_strategies("v8", declaration_fusion=True)[0]
    host = Seed(content="let o = new Object();\n")
    donor = Seed(content="function Shape(a) {\n  this.a = a;\n}\n")
    assert strategy.is_viable_pair(host, donor)
    assert "new Shape(" in strategy.fuse(host, donor).content


def test_declaration_fusion_declares_before_use():
    """`class` bindings sit in the temporal dead zone, so constructing one
    above its declaration is a ReferenceError, not the shape confusion the
    technique is aiming at."""
    strategy = get_strategies("v8", declaration_fusion=True)[0]
    host = Seed(content="let o = new Object();\n")
    donor = Seed(content="class Donor {\n  constructor() { this.x = 1; }\n}\n")
    content = strategy.fuse(host, donor).content
    assert content.index("class Donor") < content.index("new Donor(")


def test_declaration_fusion_reports_nonviable_when_nothing_to_donate():
    strategy = get_strategies("v8", declaration_fusion=True)[0]
    host = Seed(content="let o = new Object();\n")
    donor = Seed(content="let plain = 1;\n")
    assert not strategy.is_viable_pair(host, donor)


def test_state_fusion_produces_a_child():
    strategy = get_strategies("v8", state_fusion=True)[0]
    a = Seed(content="function f(x) {\n  let y = x + 1;\n  return y;\n}\nf(1);\n")
    b = Seed(content="function g(z) {\n  let w = z * 2;\n  return w;\n}\ng(2);\n")
    child = strategy.fuse(a, b)
    assert child and child.content.strip()
    assert child.metadata["type"] == "javascript"


def test_all_three_strategies_survive_random_pairs():
    """The strategies must not raise on real-world-shaped input."""
    random.seed(11)
    corpus = [
        Seed(content='// Flags: --allow-natives-syntax\nclass A { constructor(){this.p=1;} }\n'
                     'let a = new A();\n%OptimizeFunctionOnNextCall(A);\n'),
        Seed(content='"use strict";\nconst arr = new Int32Array(8);\n'
                     'for (let i = 0; i < 8; i++) arr[i] = i;\n'),
        Seed(content='function ctor(v) { this.v = v; }\nlet o = {};\no.x = 1;\n'),
        Seed(content='const shared = [];\n{\nconst shared2 = 1;\n}\nshared.push(1);\n'),
        Seed(content='let big = new ArrayBuffer(1024);\nlet dv = new DataView(big);\n'),
    ]
    for kw in (dict(dataflow_fusion=True), dict(declaration_fusion=True),
               dict(state_fusion=True)):
        strategy = get_strategies("v8", **kw)[0]
        for a in corpus:
            for b in corpus:
                if a is b:
                    continue
                if not strategy.is_viable_pair(a, b):
                    continue
                child = strategy.fuse(a, b)
                assert child is not None and child.content.strip()


# ---------------------------------------------------------------------------
# Flags that break the oracle
# ---------------------------------------------------------------------------

def test_stack_overflow_fatal_is_not_a_bug():
    """V8's `FATAL("Aborting on stack overflow")`.

    Under --correctness-fuzzer-suppressions, Isolate::StackOverflow turns
    an ordinary JavaScript stack overflow into a fatal error
    (src/execution/isolate.cc). Fused code recurses without bound
    constantly, so left unhandled this one condition would dominate the
    findings — it produced the very first "crash" this adapter reported.
    """
    output = """

#
# Fatal error in ../../src/execution/isolate.cc, line 2522
# Aborting on stack overflow
#
    v8::internal::Isolate::StackOverflow()
Received signal 6
"""
    verdict = analyzer.classify(output)
    assert verdict["is_bug"] is False, verdict
    assert verdict["kind"] == "resource"


def test_oracle_breaking_seed_flags_are_dropped():
    """A seed's own `// Flags:` line must not be able to reintroduce one."""
    import core.config_loader  # noqa: F401
    driver_mod = _load("ffl_v8_driver_test", "projects/v8/driver.py")
    cfg = {"execution": {"mem_limit_mb": 512, "heap_limit_mb": 256}}
    driver = driver_mod.V8Driver(cfg)
    facts = analyzer.analyze_seed(
        "// Flags: --correctness-fuzzer-suppressions --expose-gc\n")
    chosen = driver._choose_flags(facts)
    assert "--correctness-fuzzer-suppressions" not in chosen
    assert "--expose-gc" in chosen


def test_memory_cap_does_not_use_ulimit_v():
    """`ulimit -v` and ASan are incompatible: ASan reserves ~20 TB of
    virtual address space, so a useful -v cap kills d8 before main()."""
    driver_mod = _load("ffl_v8_driver_test2", "projects/v8/driver.py")
    driver = driver_mod.V8Driver({"execution": {"mem_limit_mb": 512}})
    cmd = driver._build_command("/tmp/x.js", {"flags": [], "uses_natives": False})
    assert "ulimit -v" not in cmd
    assert "hard_rss_limit_mb=512" in cmd


def test_deliberate_crash_primitives_are_not_bugs():
    """V8 ships primitives whose whole job is to crash on command.

    triggerAssertFalse() is literally `DCHECK(false)`
    (src/extensions/trigger-failure-extension.cc) and %AbortJS calls
    base::OS::Abort. Both arrive wearing exactly the clothes of a real
    finding — a DCHECK header, an "Aborted" signal line — so they have to
    be subtracted by shape, not by return code.
    """
    trigger = """

#
# Fatal error in ../../src/extensions/trigger-failure-extension.cc, line 49
# Debug check failed: false.
#
Received signal 6
"""
    assert analyzer.classify(trigger)["is_bug"] is False
    abortjs = "abort: expected 1 but found 2\n\n==== JS stack trace ====\nAborted\n"
    assert analyzer.classify(abortjs)["is_bug"] is False


def test_trigger_failure_flag_is_withheld():
    driver_mod = _load("ffl_v8_driver_test3", "projects/v8/driver.py")
    driver = driver_mod.V8Driver({"execution": {}})
    facts = analyzer.analyze_seed("// Flags: --expose-trigger-failure\n")
    assert "--expose-trigger-failure" not in driver._choose_flags(facts)


def test_abortjs_is_disabled_on_every_run():
    driver_mod = _load("ffl_v8_driver_test4", "projects/v8/driver.py")
    driver = driver_mod.V8Driver({"execution": {}})
    cmd = driver._build_command("/tmp/x.js", {"flags": [], "uses_natives": False})
    assert "--disable-abortjs" in cmd


def test_fusion_output_is_preserved_under_the_harness():
    """The harness wraps and appends; it must not alter what fusion made.

    The three techniques are the product here. The guard and the epilogue
    exist to get that product executed by the optimising compilers, and a
    change that quietly rewrote the fused body would defeat the point of
    measuring the techniques at all.
    """
    strategy = get_strategies("v8", dataflow_fusion=True)[0]
    a = Seed(content="function alpha(x) { return x + 1; }\nalpha(1);\n")
    b = Seed(content="function beta(y) { return y * 2; }\nbeta(2);\n")
    child = strategy.fuse(a, b)
    # Every line the strategies produced survives verbatim.
    for line in ("function alpha(x) { return x + 1; }", "alpha(1);",
                 "function beta(y) { return y * 2; }", "beta(2);"):
        assert line in child.content, child.content


def test_harness_is_not_baked_into_fusion_output():
    """Fusion output must be exactly what the techniques produced.

    The harness is applied by the driver at execution time, not written
    into the child. A fused child can be fused again, so a harness in the
    text would compound across a chain — and it would offer its own
    identifiers to the dataflow rename, which was caught renaming a
    seed's variable to the harness's `ARGS`.
    """
    for kw in (dict(dataflow_fusion=True), dict(state_fusion=True)):
        strategy = get_strategies("v8", **kw)[0]
        child = strategy.fuse(Seed(content="function u(x){return x;}\nu(1);\n"),
                              Seed(content="function v(y){return y;}\nv(2);\n"))
        for marker in ("__ffl_heat", "__ffl_pre", "FFLDIGEST",
                       "%OptimizeFunctionOnNextCall"):
            assert marker not in child.content, (kw, marker)


def test_driver_applies_the_harness_at_execution_time():
    driver_mod = _load("ffl_v8_driver_harness", "projects/v8/driver.py")
    src = "// Flags: --expose-gc\nfunction a(x) { return x; }\n"
    wrapped = driver_mod.apply_harness(src)
    # The flags line must stay first; nothing may displace it.
    assert wrapped.splitlines()[0] == "// Flags: --expose-gc"
    assert "FFLDIGEST" in wrapped
    assert "function a(x) { return x; }" in wrapped
    # The names are found lexically. A snapshot of globalThis cannot do
    # this: `function a` is hoisted, so it is already on the global object
    # before any statement runs and would be mistaken for a built-in.
    assert '__FFL_NAMES = ["a"]' in wrapped


def test_hoisted_and_bound_declarations_are_all_collected():
    driver_mod = _load("ffl_v8_driver_names", "projects/v8/driver.py")
    names = driver_mod._toplevel_callables(
        "function f(){}\n"
        "var g = function(){};\n"
        "let h = (x) => x;\n"
        "const k = x => x;\n"
        "class C {}\n"
        "  function indented(){}\n"      # not top level
        "let plain = 42;\n")             # not callable
    assert names == ["f", "g", "h", "k", "C"]


def test_harness_is_skipped_when_there_is_nothing_to_exercise():
    driver_mod = _load("ffl_v8_driver_skip", "projects/v8/driver.py")
    src = "let a = 1;\nconsole.log(a);\n"
    assert driver_mod.apply_harness(src) == src


# ---------------------------------------------------------------------------
# The `// Flags:` line is configuration, not program text
# ---------------------------------------------------------------------------

def test_mutation_does_not_touch_the_flags_line():
    """The mutator must not rewrite `// Flags:`.

    It is passed to d8 on the command line, but to a mutator it is a
    comment with integers in it. Mutating it turned `--stack-size=100`
    into `--stack-size=2147483647`, and V8 then fails a DCHECK setting up
    the stack guard — before any JavaScript runs, and reproducibly on an
    empty script. Two of the campaign's first findings were that.
    """
    strategy = get_strategies("v8", dataflow_fusion=True)[0]
    a = Seed(content="// Flags: --stack-size=100\nlet p = 1;\n")
    b = Seed(content="// Flags: --max-valid-polymorphic-map-count=4\nlet q = 2;\n")
    for _ in range(40):
        header = strategy.fuse(a, b).content.splitlines()[0]
        assert "--stack-size=100" in header, header
        assert "--max-valid-polymorphic-map-count=4" in header, header


def test_absurd_numeric_flag_values_are_rejected():
    """Defence in depth for a value that arrives some other way."""
    facts = analyzer.analyze_seed(
        "// Flags: --stack-size=2147483647 --expose-gc --stack-size=100\n")
    assert "--stack-size=2147483647" not in facts["flags"]
    assert "--expose-gc" in facts["flags"]
    assert "--stack-size=100" in facts["flags"]


# ---------------------------------------------------------------------------
# Differential (correctness) testing
# ---------------------------------------------------------------------------

def _diff_driver(**exec_cfg):
    driver_mod = _load("ffl_v8_driver_diff", "projects/v8/driver.py")
    return driver_mod.V8Driver({"execution": exec_cfg})


def test_digest_is_emitted_and_tier_independent_by_construction():
    """The digest may only contain values that are equal by definition in
    both configurations — otherwise every program is a false mismatch."""
    driver_mod = _load("ffl_v8_driver_digest", "projects/v8/driver.py")
    epi = driver_mod._JS_EPILOGUE
    # Object identity and contents never enter it.
    assert '"o:" + t' in epi
    # An error contributes its *type*; a message can carry a line number
    # or a value formatted differently by an optimised frame.
    assert '"E:" + ((e && e.name)' in epi
    # -0 and NaN are canonicalised rather than stringified.
    assert '"n:NaN"' in epi and '"n:-0"' in epi


def test_differential_flags_differ_only_in_optimization():
    """Both sides must be --predictable, or the comparison measures the
    scheduler and the GC instead of the compiler."""
    drv = _diff_driver()
    assert "--predictable" in drv.DIFF_BASE_FLAGS
    assert "--jitless" in drv.DIFF_BASE_FLAGS
    for flags in drv.DIFF_OPT_FLAGS:
        assert "--predictable" in flags
        assert "--jitless" not in flags


def test_only_the_digest_is_compared():
    """Everything else d8 prints varies with the tier legitimately.

    Comparing whole output ran for 13 hours and produced 14 mismatches,
    every one of them a difference in which uncaught exceptions were
    reported, how many "Stack overflow" lines were hit, or the script
    path in a diagnostic — not one a miscompile.
    """
    drv = _diff_driver()
    noisy = ("FFLDIGEST 12345 fns=3\n"
             "/tmp/x/test.js:9: ReferenceError: q is not defined\n"
             "Stack overflow\nStack overflow\n"
             "1 pending unhandled Promise rejection(s) detected.\n")
    assert drv._digest_of(noisy) == "12345"


def test_a_digest_with_no_functions_carries_no_signal():
    """fns=0 means the epilogue found nothing to call, so the digest is a
    constant and comparing it says nothing about the program."""
    drv = _diff_driver()
    assert drv._digest_of("FFLDIGEST 0 fns=0\n") is None
    assert drv._digest_of("no digest at all\n") is None


def test_differential_can_be_disabled():
    assert _diff_driver(differential_rate=0).diff_rate == 0
    assert _diff_driver(differential_rate=0.5).diff_rate == 0.5


def test_tier_reporting_natives_skip_the_differential_check():
    """Natives that report the compiler's own state are a guaranteed
    mismatch: reporting the tier is their purpose, and the tier is exactly
    what the two configurations differ in.

    %IsBeingInterpreted is the one this adapter missed, and it produced
    the campaign's first differential finding.
    """
    drv = _diff_driver()
    for src in ("function f(){ return %IsBeingInterpreted(); }",
                "function f(){ return %GetOptimizationStatus(f); }",
                "function f(){ return %ActiveTierIsTurbofan(f); }",
                "function f(){ return %IsTurbofanEnabled(); }",
                "function f(){ return %CurrentFrameIsTurbofan(); }",
                "function f(){ return Math.random(); }",
                "function f(){ return new Date().getTime(); }",
                "function f(){ try { null.x } catch (e) { return e.stack; } }"):
        assert drv._DIFF_INELIGIBLE.search(src), src


def test_ordinary_programs_remain_differential_eligible():
    drv = _diff_driver()
    for src in ("function f(x){ return x * 2; }",
                "function g(a){ let s = 0; for (const v of a) s += v; return s; }",
                "function h(){ return %OptimizeFunctionOnNextCall(h); }"):
        assert not drv._DIFF_INELIGIBLE.search(src), src
