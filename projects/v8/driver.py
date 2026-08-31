"""
projects/v8/driver.py — run a fused JavaScript program under d8 and report
whether what came back is a bug.

This file owns execution: writing the script, choosing flags, containing
the run, cleaning up. Judgement lives in projects/v8/analyzer.py.

Why the flags carry so much weight here
---------------------------------------
V8 is not a compiler that reads a file and stops. The same script takes a
completely different path through the engine depending on which tier
compiles it, when the GC runs, and which experimental pipeline is on. A
fused program executed once under default flags exercises the interpreter
and almost nothing else — the optimising compilers never even see a
function that runs a handful of times.

So the flags are how this adapter reaches the code where V8's bugs
actually live, and they fall into groups:

  Tiering        Force the function through Maglev or TurboFan instead of
                 waiting for it to get hot: --stress-maglev, --max-opt,
                 --minimum-invocations-before-optimization=N, and the
                 negations (--no-turbofan, --no-maglev, --jitless) that
                 pin it to a lower tier. A miscompile only exists if the
                 optimising compiler runs.

  Deopt          --deopt-every-n-times=N forces the optimised-to-
                 interpreter transition repeatedly. Deoptimisation is
                 where the JIT's assumptions get reconciled with reality,
                 and it is historically dense with bugs.

  GC             --stress-scavenge, --stress-compaction,
                 --stress-incremental-marking, --gc-interval=N,
                 --compact-on-every-full-gc. These move objects at moments
                 the code did not expect, which is how a missing write
                 barrier or a stale pointer becomes visible.

  Verification   --verify-heap, --verify-csa, --turbo-verify,
                 --assert-types, --maglev-assert. Checks that run *during*
                 execution rather than at build time. They are the runtime
                 half of the oracle the debug build gives us.

  New pipelines  --turboshaft, --turbolev, --future, --harmony. Newer code
                 is less exercised code.

The unsafe surface
------------------
`--allow-natives-syntax` enables V8's %-prefixed runtime functions —
%OptimizeFunctionOnNextCall, %DebugPrint, %CollectGarbage and the rest.
These are the engine's internals exposed directly to script, with none of
the checking that guards ordinary JavaScript, and they are the closest
JavaScript analogue to Rust's `unsafe`: they let a script drive V8 into
states the language alone cannot reach.

It is also not optional for much of the corpus. mjsunit tests use natives
syntax heavily, and `%Foo()` is a *syntax* error without the flag — so a
seed containing one is unrunnable unless the flag is present. The seed
analysis in analyzer.py detects this and the flag is forced on.

The tradeoff is that %-functions can legitimately crash when misused, and
fusion misuses them constantly. Two things keep that from flooding the
oracle: --fuzzing (which V8 provides precisely so fuzzers can use natives
syntax — it makes the runtime functions tolerate bad arguments instead of
CHECK-failing) and --correctness-fuzzer-suppressions.
"""

import os
import random
import re
import shutil
import time

from core.driver import BaseDriver, ExecutionResult

# core/driver.py's get_driver loads this file by path, so there is no
# parent package for a relative import to resolve against.
try:
    from projects.v8.analyzer import analyze_seed, classify, is_resource_exhaustion
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_v8_analyzer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed = _analyzer.analyze_seed
    classify = _analyzer.classify
    is_resource_exhaustion = _analyzer.is_resource_exhaustion


# Declarations the harness should exercise, found lexically in the source.
#
# A snapshot of `globalThis` taken before the body runs cannot be used to
# tell the seed's functions from V8's built-ins: `function f(){}` is
# *hoisted*, so it is already on the global object before the first
# statement executes and lands in the snapshot alongside the built-ins.
# That silently excluded exactly the functions worth heating.
_JS_TOPLEVEL_FN_RE = re.compile(
    r'^(?:function\s*\*?\s*([A-Za-z_$][\w$]*)'
    r'|(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?'
    r'(?:function|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)'
    r'|class\s+([A-Za-z_$][\w$]*))', re.M)


def _toplevel_callables(source):
    """Names declared at column zero that the epilogue can call."""
    names = []
    for m in _JS_TOPLEVEL_FN_RE.finditer(source):
        name = m.group(1) or m.group(2) or m.group(3)
        if name and name not in names:
            names.append(name)
    return names


# Runs after them. Calls each function the seeds defined, repeatedly and
# with arguments of mixed type, then asks V8 to optimise it.
#
# Why this exists
# ---------------
# V8's bugs live in the optimising compilers, and a compiler only runs on
# a function that got hot. mjsunit seeds mostly *define* functions and
# call them two or three times, so a fused program left alone stays in the
# interpreter: measured on 80 fused pairs, 11% reached an optimising tier.
# With this epilogue — and only in combination with the guards below —
# that goes to 69%.
#
# The mixed-type arguments are the point, not padding: feeding one call
# site an int, then a double, then a string drives its inline caches
# polymorphic and then megamorphic, which is the state the optimiser has
# to reason about and the state its bugs live in.
#
# Every call is individually wrapped: these functions are being handed
# arguments they were never written for, so most calls throw, and one
# throw must not stop the rest from running.
_JS_EPILOGUE = """\
;(function __ffl_heat() {
  var ARGS = [0, 1, -1, 1.5, NaN, Infinity, "s", true, null, undefined,
              {}, [], 2 ** 31, -(2 ** 31), 9007199254740993n, Symbol("s")];
  var fns = [];
  for (var i = 0; i < __FFL_NAMES.length; i++) {
    var v;
    // globalThis for `function` and `var` (both hoisted out of the guard
    // block); eval reaches anything else still in scope here.
    try { v = globalThis[__FFL_NAMES[i]]; } catch (e) {}
    if (typeof v !== "function") {
      try { v = eval(__FFL_NAMES[i]); } catch (e) { continue; }
    }
    if (typeof v === "function") fns.push(v);
  }

  // Running digest of everything the seeds' functions returned. Compared
  // between an unoptimised and an optimised run of the same program: if
  // they disagree, one of the two compiled the code wrongly. Only values
  // that are the same *by definition* in both runs may go in, so this is
  // deliberately lossy — see __ffl_val.
  var d = 0;
  function __ffl_tok(s) {
    for (var i = 0; i < s.length; i++) d = (d * 31 + s.charCodeAt(i)) | 0;
  }
  function __ffl_val(v) {
    var t = typeof v;
    if (t === "number") {
      if (v !== v) return "n:NaN";                  // NaN !== NaN
      if (v === 0) return 1 / v === -Infinity ? "n:-0" : "n:0";
      return "n:" + v;
    }
    if (t === "string") {                           // hash, not identity
      var h = 0;
      for (var i = 0; i < v.length; i++) h = (h * 31 + v.charCodeAt(i)) | 0;
      return "s:" + v.length + ":" + h;
    }
    if (t === "bigint") return "b:" + v;
    if (t === "boolean" || t === "undefined") return t + ":" + v;
    if (v === null) return "null";
    if (t === "symbol") return "sym";
    // Never the object's contents or identity: an address differs between
    // runs for reasons that have nothing to do with compilation.
    return "o:" + t;
  }

  for (var j = 0; j < fns.length; j++) {
    var f = fns[j];
    try { %PrepareFunctionForOptimization(f); } catch (e) {}
    for (var n = 0; n < 12; n++) {
      for (var m = 0; m < ARGS.length; m++) {
        try { __ffl_tok(__ffl_val(f(ARGS[m], ARGS[m], ARGS[m]))); }
        // The *type* of the failure only. A message can carry a line
        // number or a value formatted differently by an optimised frame,
        // and that difference is not a miscompile.
        catch (e) { __ffl_tok("E:" + ((e && e.name) ? e.name : "?")); }
      }
    }
    try { %OptimizeFunctionOnNextCall(f); } catch (e) {}
    for (var m2 = 0; m2 < ARGS.length; m2++) {
      try { __ffl_tok(__ffl_val(f(ARGS[m2], ARGS[m2], ARGS[m2]))); }
      catch (e) { __ffl_tok("E:" + ((e && e.name) ? e.name : "?")); }
    }
  }
  console.log("FFLDIGEST " + d + " fns=" + fns.length);
})();"""


def _js_guard(body):
    """Wrap a body so an exception in it does not kill the whole script.

    Fusing two unrelated programs produces an uncaught exception almost
    every time — measured at 91% of fused pairs — and in JavaScript that
    terminates the script. Everything after the throw, including the other
    seed's half and the epilogue above, never runs: of 40 fused programs
    with a probe appended, the probe was reached exactly once.

    Guarding alone changes nothing about how far into the engine the code
    gets (measured: 11% before, 11% after) — it only matters together with
    the epilogue, which it lets run at all. Neither half of the pair is
    useful on its own.
    """
    return "try {\n" + body + "\n} catch (e) {}"


# Whether assemble_js_file appends the optimisation epilogue.
#
# The guard is not controversial — it only wraps what fusion produced, the
# way assemble_go_file adds the `package` clause Go requires. The epilogue
# is different in kind: it injects ~20 lines that *no fusion strategy
# produced*, so the program under test stops being purely fusion-derived.
# It does not change any of the three techniques — dataflow renaming,
# state interleaving and declaration injection all run exactly as before,
# and their output is what the epilogue then exercises — but it does
# change what reaches the engine.
#
# It is on by default because it is what makes V8's optimising compilers
# run at all (measured: 11% of fused programs reached an optimising tier
# without it, ~24% with it, and V8's bugs live in those tiers). Set
# FFL_V8_HEAT=0 to get fusion output alone.

# Whether to wrap the program in the harness above at execution time.
#
# The guard only wraps what fusion produced; the epilogue injects code no
# fusion strategy produced, so the program that reaches d8 is no longer
# purely fusion-derived. It is on by default because without it 89% of
# fused programs never reach an optimising compiler at all (measured: 11%
# with neither, 11% with the guard alone, 14% with the epilogue alone
# because it almost never runs, ~24% with both). Set FFL_V8_HEAT=0 for
# fusion output alone.
JS_HEAT = os.environ.get("FFL_V8_HEAT", "1") != "0"

# The epilogue's last line. It is the only part of the output that is
# tier-independent by construction, and therefore the only part the
# differential comparison may look at.
DIGEST_RE = re.compile(r"^FFLDIGEST (-?\d+) fns=(\d+)$", re.M)


def apply_harness(source):
    """Wrap a fused program for execution.

    Splits off the leading `// Flags:` and directive lines so nothing
    displaces them: a directive prologue is only a directive when it is
    the first statement.
    """
    if not JS_HEAT:
        return source
    head, body = [], []
    for line in source.splitlines():
        stripped = line.strip()
        if not body and (stripped.startswith("// Flags:")
                         or stripped in ('"use strict";', "'use strict';",
                                         '"use strict"', "'use strict'")):
            head.append(line)
        else:
            body.append(line)
    text = "\n".join(body)
    names = _toplevel_callables(text)
    if not names:
        return source          # nothing to exercise; leave it alone
    decl = "var __FFL_NAMES = [" + ",".join(
        '"%s"' % n for n in names[:200]) + "];"
    return "\n".join(head + [decl, _js_guard(text), _JS_EPILOGUE]) + "\n"


class V8Driver(BaseDriver):
    """Drives the d8 shell built by projects/v8/setup.py."""

    # Every flag below was checked against src/flags/flag-definitions.h in
    # the pinned checkout. d8 rejects an unknown flag and fails the whole
    # execution, so an unverified name silently costs every run that draws
    # it. (--stress-opt and --always-turbofan, which older V8 fuzzing
    # documentation recommends, no longer exist and are deliberately absent.)

    # Force a function into an optimising tier rather than waiting for it
    # to become hot. Without one of these the optimising compilers never
    # run at all on a short fused script.
    TIER_FLAGS = [
        "--max-opt=3",
        "--stress-maglev",
        "--minimum-invocations-before-optimization=1",
        "--minimum-invocations-before-optimization=2",
        "--always-sparkplug",
        "--stress-background-compile",
        "--concurrent-recompilation",
        "--stress-concurrent-inlining",
        "--turboshaft",
        "--turbolev",
    ]

    # Pin execution to a *lower* tier, or turn a tier off. A difference in
    # behaviour between tiers is the miscompile; reaching only the fast
    # path would hide half of them.
    TIER_DOWN_FLAGS = [
        "--jitless",
        "--no-turbofan",
        "--no-maglev",
        "--no-sparkplug",
        "--no-use-ic",
        "--no-lazy-feedback-allocation",
        "--force-slow-path",
        "--max-opt=0",
        "--max-opt=1",
        "--max-opt=2",
    ]

    # Runtime verification: the half of the oracle that runs during
    # execution rather than being compiled in. Expensive, hence sampled.
    VERIFY_FLAGS = [
        "--verify-heap",
        "--verify-csa",
        "--turbo-verify",
        "--turbo-verify-machine-graph=*",
        "--assert-types",
        "--maglev-assert",
        "--verify-predictable",
        "--stress-lazy-source-positions",
    ]

    # Move objects when the code does not expect it. This is how a missing
    # write barrier or a stale pointer stops being invisible.
    GC_FLAGS = [
        "--stress-scavenge=100",
        "--stress-compaction",
        "--stress-incremental-marking",
        "--stress-concurrent-allocation",
        "--compact-on-every-full-gc",
        "--gc-interval=100",
        "--gc-interval=500",
        "--stress-flush-code",
        "--minor-ms",
        "--stress-per-context-marking-worklist",
        "--expose-gc",
    ]

    # Deoptimisation: reconciling the optimising compiler's assumptions
    # with reality, and historically dense with bugs.
    DEOPT_FLAGS = [
        "--deopt-every-n-times=1",
        "--deopt-every-n-times=7",
        "--deopt-every-n-times=53",
    ]

    # Determinism and threading. --predictable serialises everything, which
    # both makes a finding reproducible and exercises a different scheduler
    # path than the concurrent default.
    EXEC_MODE_FLAGS = [
        "--predictable",
        "--single-threaded",
        "--stress-snapshot",
        "--future",
        "--harmony",
        "--shared-string-table",
        "--expose-externalize-string",
    ]

    # Always present. --fuzzing is V8's own switch for exactly this
    # situation: it makes the %-runtime functions tolerate the garbage
    # arguments a fuzzer inevitably passes, instead of CHECK-failing and
    # burying the real findings.
    #
    # --correctness-fuzzer-suppressions is deliberately NOT here, despite
    # its name. It belongs to V8's *correctness* fuzzer, which compares
    # output between two configurations and for which a stack overflow
    # makes the comparison meaningless — so under that flag
    # Isolate::StackOverflow does `FATAL("Aborting on stack overflow")`
    # (src/execution/isolate.cc). For a crash-finding fuzzer that
    # manufactures a fatal error out of an ordinary JavaScript condition
    # that fused code hits constantly. It was in this list for exactly one
    # run and produced exactly one finding, which was that.
    # --disable-abortjs turns %AbortJS from an abort into a printed line
    # (src/runtime/runtime-test.cc). mjsunit's own assertion helpers call
    # it, so without this every fused program that trips an assertion
    # written *by the test* looks like an engine abort.
    BASE_FLAGS = ["--fuzzing", "--disable-abortjs"]

    # Never passed, wherever they come from — including a seed's own
    # `// Flags:` line, which for an mjsunit test written for V8's
    # correctness fuzzer will carry the first two.
    ORACLE_BREAKING_FLAGS = frozenset({
        # Turns an ordinary JS stack overflow into FATAL. See BASE_FLAGS.
        "--correctness-fuzzer-suppressions",
        # Exposes triggerAssertFalse()/triggerCheckFalse(), whose entire
        # implementation is `DCHECK(false)` / `CHECK(false)`
        # (src/extensions/trigger-failure-extension.cc). V8 uses them to
        # self-test its assertion machinery; for this fuzzer they are a
        # guaranteed finding that says nothing.
        "--expose-trigger-failure",
        # A DEBUG-only budget check: DCHECK_LT(before, 49152) on the live
        # handle count (src/handles/handles-inl.h). The epilogue calls
        # every function in the program a few hundred times and passes it
        # legitimately. It asserts a budget, not a correctness invariant.
        "--check-handle-count",
        # Suppresses the very output the oracle reads.
        "--no-abort-on-contradictory-flags",
        "--hard-abort",
        # Would write the engine's state to a path from the seed's original
        # directory, which no longer exists.
        "--redirect-code-traces",
    })

    # A fused script can allocate without bound. Without a cap the host OOM
    # killer fires and can take the orchestrator with it; with one, V8
    # reports OOM and exits, which analyzer.classify files as resource
    # exhaustion rather than a bug.
    #
    # Enforced through ASAN_OPTIONS=hard_rss_limit_mb, NOT `ulimit -v`.
    # ASan reserves ~20 TB of virtual address space for its shadow memory
    # at startup, so any `ulimit -v` small enough to be a useful cap kills
    # the process before main() with "ReserveShadowMemoryRange failed" —
    # every execution fails identically and the run reports a 0% valid
    # rate. hard_rss_limit_mb bounds resident memory instead, which is the
    # quantity that actually threatens the host.
    DEFAULT_MEM_LIMIT_MB = 4096
    # V8's own heap ceiling, kept below the address-space cap so the engine
    # reports a clean "JavaScript heap out of memory" before the ulimit
    # turns it into a bare allocation failure.
    DEFAULT_HEAP_MB = 1024

    # ------------------------------------------------------------------
    # Differential (correctness) testing
    # ------------------------------------------------------------------
    # A miscompile does not crash — it computes the wrong answer. That is
    # V8's most productive bug class and it is completely invisible to a
    # crash oracle. The only way to see it is to run the same program
    # twice under configurations that must agree by definition, and
    # compare.
    #
    # The two configurations differ in exactly one thing: whether the
    # optimising compilers run. Anything else that changed would give a
    # legitimate difference and a false positive.
    #
    # --predictable pins the scheduler and the GC so the two runs are
    # comparable at all; it measured 0 nondeterministic results out of 34
    # programs that produce output.
    DIFF_BASE_FLAGS = ["--predictable", "--jitless"]
    DIFF_OPT_FLAGS = [
        ["--predictable", "--max-opt=3", "--minimum-invocations-before-optimization=1"],
        ["--predictable", "--stress-maglev"],
        ["--predictable", "--max-opt=3", "--stress-concurrent-inlining"],
        ["--predictable", "--max-opt=3", "--no-lazy-feedback-allocation"],
    ]
    # Fraction of executions spent on differential testing. Each costs two
    # runs (three on a mismatch, for the confirmation), so this trades
    # crash-fuzzing throughput directly.
    DEFAULT_DIFF_RATE = 0.30

    # Programs that cannot take part in a differential comparison because
    # they would differ between the two configurations *legitimately*.
    #
    # The natives here report the compiler's own state — which tier is
    # running, whether the frame is optimised, whether a tier is even
    # enabled. That is their entire purpose, and it is exactly what the
    # two configurations differ in, so each one is a guaranteed mismatch.
    # The list is taken from V8's runtime function definitions rather than
    # written from memory; %IsBeingInterpreted was the one this adapter
    # missed, and it produced the campaign's first differential finding.
    #
    # These programs are still fuzzed for crashes — only the correctness
    # comparison is skipped. Dropping them from the corpus instead would
    # give up the crash coverage too.
    _DIFF_INELIGIBLE = re.compile(
        r"%(?:ActiveTierIs\w+"
        r"|CurrentFrameIsTurbofan"
        r"|GetOptimizationStatus"
        r"|IsBeingInterpreted"
        r"|IsConcurrentRecompilationSupported"
        r"|IsTurbofanEnabled|IsMaglevEnabled|IsSparkplugEnabled"
        r"|IsDictPropertyConstTrackingEnabled"
        r"|TurbofanStaticAssert"
        r"|WasmDeoptsExecuted\w*"
        r"|DebugPrint|GetUndetectable)\s*\("
        # Nondeterministic between any two runs, tiers aside.
        r"|\b(?:Math\.random|Date\.now|performance\.now)\s*\("
        r"|\bnew\s+Date\s*\("
        # Stack text names frames, and an optimised frame differs.
        r"|\.stack\b")

    # Lines that legitimately differ between the two configurations, or
    # between any two runs, and must be dropped before comparing.
    _DIFF_VOLATILE = re.compile(
        r"^.*(?:"
        r"0x[0-9a-f]{4,}"                 # any address
        r"|\bat [\w$.<>]+ \("            # stack frame lines
        r"|V8 is running with"            # the natives-syntax banner
        r"|optimization status"           # %GetOptimizationStatus output
        r").*$", re.M | re.I)

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.mem_limit_mb = int(exec_cfg.get("mem_limit_mb",
                                             self.DEFAULT_MEM_LIMIT_MB))
        self.heap_mb = int(exec_cfg.get("heap_limit_mb", self.DEFAULT_HEAP_MB))
        self.d8 = os.path.join(self.ffl_root, "projects", "v8", "v8-src", "v8",
                               "out", "fuzz", "d8")
        self.diff_rate = float(exec_cfg.get("differential_rate",
                                            self.DEFAULT_DIFF_RATE))

    # -- flag selection ----------------------------------------------------

    def _choose_flags(self, facts):
        """Assemble one execution's flag set.

        Draws one group at a time rather than sampling a flat list, so a
        run reliably gets a tier decision *and* a GC decision instead of
        three GC flags and nothing else.
        """
        flags = list(self.BASE_FLAGS)
        flags.append(f"--max-old-space-size={self.heap_mb}")

        # Unconditional. %-syntax is a *parse* error without this, and
        # every fused program carries the optimisation epilogue that
        # core/fusion.assemble_js_file appends, which calls
        # %PrepareFunctionForOptimization and %OptimizeFunctionOnNextCall.
        # Omitting the flag would not merely skip the epilogue — the whole
        # script would fail to parse.
        #
        # It is also the surface worth reaching: the %-runtime functions
        # are V8's internals exposed to script with none of the checking
        # that guards ordinary JavaScript.
        flags.append("--allow-natives-syntax")

        # The seed's own `// Flags:` line: part of what the test exercises,
        # minus the ones that would break the oracle rather than steer the
        # engine. See OracleBreakingFlags for why each is listed.
        flags.extend(f for f in (facts.get("flags") or [])
                     if f.split("=", 1)[0] not in self.ORACLE_BREAKING_FLAGS)

        # Tier. Mostly upward (a miscompile requires the optimiser to have
        # run), sometimes downward to reach the slow paths.
        r = random.random()
        if r < 0.55:
            flags.append(random.choice(self.TIER_FLAGS))
        elif r < 0.75:
            flags.append(random.choice(self.TIER_DOWN_FLAGS))

        if random.random() < 0.35:
            flags.append(random.choice(self.DEOPT_FLAGS))
        if random.random() < 0.40:
            flags.append(random.choice(self.GC_FLAGS))
        # Verification is the expensive half; one at a time.
        if random.random() < 0.45:
            flags.append(random.choice(self.VERIFY_FLAGS))
        if random.random() < 0.25:
            flags.append(random.choice(self.EXEC_MODE_FLAGS))

        # Deduplicate while keeping order: a seed's own `// Flags:` may
        # repeat one we drew, and d8 warns about a flag given twice.
        seen, out = set(), []
        for f in flags:
            key = f.split("=", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            out.append(f)
        return out

    def _sanitizer_env(self):
        """ASan/UBSan settings, as a shell prefix.

        hard_rss_limit_mb   the memory cap; see DEFAULT_MEM_LIMIT_MB for
                            why this and not `ulimit -v`.
        allocator_may_return_null=1
                            a huge allocation returns null instead of
                            aborting with an ASan report. Fused scripts ask
                            for absurd array lengths constantly, and
                            without this every one of them is a "crash"
                            that has nothing to do with a V8 bug.
        symbolize=1         an unsymbolised report is addresses, which the
                            signature would key on and never deduplicate.
        detect_leaks=0      the build sets is_lsan=false; a script that
                            exits with live objects is not a finding.
        handle_abort/segv   let ASan produce the report for a signal
                            instead of the process dying silently.
        print_summary=1     analyzer.py keys the signature on the SUMMARY
                            line.
        """
        asan = ":".join([
            f"hard_rss_limit_mb={self.mem_limit_mb}" if self.mem_limit_mb > 0 else "",
            "allocator_may_return_null=1",
            "symbolize=1",
            "detect_leaks=0",
            "handle_abort=1",
            "handle_segv=1",
            "print_summary=1",
            "exitcode=1",
        ]).strip(":")
        ubsan = "print_stacktrace=1:halt_on_error=0"
        return f"ASAN_OPTIONS={asan} UBSAN_OPTIONS={ubsan}"

    def _build_command(self, script, facts):
        flags = " ".join(self._choose_flags(facts))
        return f"{self._sanitizer_env()} {self.d8} {flags} {script}"

    # -- differential comparison ------------------------------------------

    def _diff_command(self, script, flags):
        joined = " ".join(self.BASE_FLAGS + ["--allow-natives-syntax",
                                             f"--max-old-space-size={self.heap_mb}"]
                          + flags)
        return f"{self._sanitizer_env()} {self.d8} {joined} {script}"

    def _digest_of(self, text):
        """The digest line, or None if the program produced no signal.

        Only the digest is compared. Everything else d8 prints varies with
        the tier for reasons that are not miscompiles: which uncaught
        exceptions got as far as being reported, how many times "Stack
        overflow" was hit (an optimised frame is a different size), the
        script path in a diagnostic, promise-rejection notices. Comparing
        whole output produced 14 mismatches in 13 hours and every one of
        them was one of those.

        fns=0 means the epilogue found no function to call, so the digest
        is a constant carrying no information about the program.
        """
        m = DIGEST_RE.search(text or "")
        if not m or m.group(2) == "0":
            return None
        return m.group(1)

    def _run_differential(self, script, workdir):
        """Run unoptimised vs optimised and compare.

        Returns (signature, detail) on a confirmed mismatch, else
        (None, None). A mismatch is re-run before being reported: a
        difference that does not reproduce is nondeterminism, not a
        miscompile, and reporting it once is how a fuzzer fills its bug
        list with noise.
        """
        opt_flags = random.choice(self.DIFF_OPT_FLAGS)

        def once(flags):
            rc, out, err = self._run_command(self._diff_command(script, flags),
                                             cwd=workdir)
            # A crash in either run is the crash oracle's business, not
            # this one; and a timeout gives no comparable output.
            if rc not in (0, 1):
                return None
            combined = f"{out}\n{err}"
            if classify(combined)["kind"] not in ("clean", "exception"):
                return None
            return self._digest_of(out)

        base = once(self.DIFF_BASE_FLAGS)
        if base is None:
            return None, None          # no comparable signal
        opt = once(opt_flags)
        if opt is None or opt == base:
            return None, None

        # Confirm. Both sides again, so a one-off difference is discarded.
        if once(self.DIFF_BASE_FLAGS) != base or once(opt_flags) != opt:
            return None, None

        detail = (f"jitless vs {' '.join(opt_flags)}\n"
                  f"digest jitless   = {base}\n"
                  f"digest optimized = {opt}")
        # Group by which optimisation configuration disagreed, not by the
        # digest values, which are unique to each program.
        return f"CORRECTNESS: {' '.join(opt_flags)}", detail

    # -- execution ---------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        diff_sig = diff_detail = None
        try:
            facts = analyze_seed(seed.content)
            script = os.path.join(workdir, "test.js")
            with open(script, "w", encoding="utf-8") as f:
                f.write(apply_harness(seed.content))
            cmd = self._build_command(script, facts)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)

            # Only on a run that neither crashed nor was killed: a program
            # that already found a crash does not also need checking for a
            # wrong answer.
            if (self.diff_rate > 0 and random.random() < self.diff_rate
                    and not self._DIFF_INELIGIBLE.search(seed.content or "")
                    and not classify(f"{stdout}\n{stderr}")["is_bug"]):
                diff_sig, diff_detail = self._run_differential(script, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        output = f"{stdout}\n{stderr}"
        verdict = classify(output)
        signature = verdict["signature"]
        crashed = verdict["is_bug"]
        if diff_sig and not crashed:
            signature, crashed = diff_sig, True
            stdout = f"{stdout}\n\n=== DIFFERENTIAL MISMATCH ===\n{diff_detail}"

        result = ExecutionResult(
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            time=time.time() - start,
            crashed=crashed,
            signature=signature,
        )
        result.command = cmd
        return result
