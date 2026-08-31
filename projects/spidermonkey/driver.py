"""
projects/spidermonkey/driver.py — run a fused JavaScript program under the
SpiderMonkey shell and report whether what came back is a bug.

This file owns execution: writing the script, choosing flags, containing
the run, cleaning up. Judgement lives in
projects/spidermonkey/analyzer.py.

Reaching the JIT
----------------
The engine's bugs live in its optimising tiers, and a tier only runs on
code that got hot. This was measured on V8, whose situation is the same:
of 80 fused programs, 11% reached an optimising compiler on their own, and
91% died on an uncaught exception before getting far enough to matter.

SpiderMonkey makes the first half of that much easier than V8 does.
`--ion-eager` compiles a function with the top-tier optimising compiler on
its first call, so no warm-up loop and no equivalent of V8's
%OptimizeFunctionOnNextCall is needed — the flag alone does it. What it
cannot fix is the second half: an uncaught exception still terminates the
script, so most of the program never runs at all. That is what the guard
in apply_harness is for.

The result is a much smaller harness than V8 needed: a try/catch around
the body and a small loop that calls what the seeds declared, with no
digest and no natives syntax.

The flags that matter
---------------------
  --ion-eager / --blinterp-eager / --baseline-eager
      Force a function into a compiler tier immediately rather than
      waiting for it to become hot. A miscompile only exists if the
      compiler ran.

  --no-ion / --no-baseline / --no-jit-backend
      Pin execution to a lower tier. A difference in behaviour between
      tiers is the bug; reaching only the fast path would hide half of
      them.

  --gc-zeal=N
      SpiderMonkey's GC stress mode: collect at moments the code did not
      expect, which is how a missing barrier or a stale pointer stops
      being invisible. This is the engine's counterpart of V8's
      --stress-scavenge and friends, and it is more direct: the modes are
      named after the hazard they provoke.

  --ion-check-range-analysis / --ion-extra-checks
      Verification that runs *during* compilation. Together with the
      MOZ_ASSERTs the debug build already carries, this is the runtime
      half of the oracle.

Every flag below was checked against js/src/shell/js.cpp in the pinned
checkout: the shell rejects an unknown option and fails the whole
execution, so an unverified name silently costs every run that draws it.
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
    from projects.spidermonkey.analyzer import analyze_seed, classify
except ImportError:  # pragma: no cover - direct-load fallback
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "ffl_sm_analyzer", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "analyzer.py"))
    _analyzer = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_analyzer)
    analyze_seed, classify = _analyzer.analyze_seed, _analyzer.classify


# ---------------------------------------------------------------------------
# Execution harness
# ---------------------------------------------------------------------------

# Declarations the harness should exercise, found lexically in the source.
#
# Lexically, and not by diffing `globalThis` before and after: `function
# f(){}` is *hoisted*, so it is already on the global object before the
# first statement runs and a before/after snapshot silently excludes
# exactly the functions worth calling. That defect went unnoticed in the
# V8 adapter until a positive control failed.
_TOPLEVEL_FN_RE = re.compile(
    r'^(?:function\s*\*?\s*([A-Za-z_$][\w$]*)'
    r'|(?:var|let|const)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s+)?'
    r'(?:function|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)'
    r'|class\s+([A-Za-z_$][\w$]*))', re.M)

# Calls what the seeds declared, with arguments of mixed type.
#
# The mixed types are the point, not padding: feeding one call site an
# int, then a double, then a string drives its inline caches polymorphic
# and then megamorphic, which is the state the optimiser has to reason
# about. Under --ion-eager the first call already compiles the function,
# so no warm-up count is needed.
#
# Every call is individually wrapped: these functions are being handed
# arguments they were never written for, so most calls throw, and one
# throw must not stop the rest.
_EPILOGUE = """\
;(function __ffl_exercise() {
  var ARGS = [0, 1, -1, 1.5, NaN, Infinity, "s", true, null, undefined,
              {}, [], 2 ** 31, -(2 ** 31), 9007199254740993n, Symbol("s")];
  for (var i = 0; i < __FFL_NAMES.length; i++) {
    var f;
    // globalThis for `function` and `var` (both hoisted out of the guard
    // block); eval reaches anything else still in scope here.
    try { f = globalThis[__FFL_NAMES[i]]; } catch (e) {}
    if (typeof f !== "function") {
      try { f = eval(__FFL_NAMES[i]); } catch (e) { continue; }
    }
    if (typeof f !== "function") continue;
    for (var n = 0; n < 3; n++) {
      for (var m = 0; m < ARGS.length; m++) {
        try { f(ARGS[m], ARGS[m], ARGS[m]); } catch (e) {}
      }
    }
  }
})();"""

# Whether to wrap the program in the harness at execution time.
#
# The guard only wraps what fusion produced; the epilogue injects code no
# fusion strategy produced, so the program that reaches the shell is no
# longer purely fusion-derived. It is on by default because without it
# most fused programs die on their first uncaught exception and never
# reach a compiler at all. Set FFL_SM_HEAT=0 for fusion output alone.
JS_HEAT = os.environ.get("FFL_SM_HEAT", "1") != "0"


# Arms the shell's own watchdog over the whole program.
#
# Two things make a fused program run long, and this covers both. The
# epilogue calls functions with deliberately huge arguments (2**31 is the
# int32 boundary, exactly where the JIT's overflow paths are interesting),
# so a function whose loop bound is its argument runs for a very long
# time. And the guard itself lets execution continue past an exception
# into code that would never otherwise have been reached — including
# loops. Arming only around the epilogue left the second case uncovered.
#
# 8 seconds, well inside the driver's own 20s subprocess timeout: the
# shell terminates cleanly with a recognisable message, which
# analyzer.py classifies as resource exhaustion, rather than being
# SIGKILLed with nothing to report.
_WATCHDOG = "try { timeout(8); } catch (e) {}"


def _guard(body):
    """Wrap a body so an exception in it does not kill the whole script.

    Fusing two unrelated programs produces an uncaught exception almost
    every time, and in JavaScript that terminates the script: everything
    after the throw, including the other seed's half and the epilogue,
    never runs.
    """
    return "try {\n" + body + "\n} catch (e) {}"


def _toplevel_callables(source):
    """Names declared at column zero that the epilogue can call."""
    names = []
    for m in _TOPLEVEL_FN_RE.finditer(source):
        name = m.group(1) or m.group(2) or m.group(3)
        if name and name not in names:
            names.append(name)
    return names


def apply_harness(source):
    """Wrap a fused program for execution.

    Splits off the leading `// |jit-test|` directive and any directive
    prologue so nothing displaces them: a `"use strict"` is only a
    directive when it is the first statement, and the jit-test line is
    read back by analyze_seed.
    """
    if not JS_HEAT:
        return source
    head, body = [], []
    for line in source.splitlines():
        stripped = line.strip()
        if not body and (stripped.startswith("// |jit-test|")
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
    return "\n".join(head + [_WATCHDOG, decl, _guard(text), _EPILOGUE]) + "\n"


class SpiderMonkeyDriver(BaseDriver):
    """Drives the shell built by projects/spidermonkey/setup.py."""

    # Force a function into a compiler tier rather than waiting for it to
    # become hot. Without one of these the optimising compilers never run
    # on a short fused script.
    TIER_FLAGS = [
        "--ion-eager",
        "--baseline-eager",
        "--blinterp-eager",
        "--fast-warmup",
        "--ion-eager --ion-offthread-compile=off",
    ]

    # Pin execution to a lower tier, or turn a tier off.
    TIER_DOWN_FLAGS = [
        "--no-ion",
        "--no-baseline",
        "--no-jit-backend",
        "--no-native-regexp",
        "--no-threads",
    ]

    # Verification that runs during compilation — the runtime half of the
    # oracle the debug build gives us.
    VERIFY_FLAGS = [
        "--ion-check-range-analysis",
        "--ion-extra-checks",
        "--spectre-mitigations=on",
    ]

    # SpiderMonkey's GC stress modes. Unlike V8's, these are numbered and
    # each names a specific hazard; 2 is "collect every N allocations",
    # 4/11 verify pre/post write barriers, 14 is compacting GC.
    GC_FLAGS = [
        "--gc-zeal=2,10",
        "--gc-zeal=4",
        "--gc-zeal=11",
        "--gc-zeal=14",
        "--gc-zeal=7",
        "--more-compartments",
        # Value-taking, not a switch: js/src/shell/js.cpp declares it with
        # addStringOption("nursery-strings", "on/off"), and passing it bare
        # makes the shell MOZ_CRASH while parsing its arguments — before a
        # line of JavaScript runs. Checking that a flag *exists* is not
        # enough; its arity has to be checked too.
        "--nursery-strings=on",
        "--nursery-strings=off",
    ]

    # Always present. --fuzzing-safe disables the shell functions that let
    # a script read files or spawn processes; the corpus calls them, and
    # on fused input those calls are neither meaningful nor safe.
    BASE_FLAGS = ["--fuzzing-safe", "--disable-oom-functions"]

    # Never passed, wherever they come from — including a seed's own
    # `// |jit-test|` line.
    ORACLE_BREAKING_FLAGS = frozenset({
        # Would make the shell exit before running anything.
        "--help", "--version",
        # Turn the artificial-OOM machinery on, which manufactures
        # allocation failures the oracle would then have to un-attribute.
        "--enable-oom-breakpoint",
    })

    # A fused script can allocate without bound. Enforced through ASan's
    # hard_rss_limit_mb, NOT `ulimit -v`: ASan reserves ~20 TB of virtual
    # address space for shadow memory at startup, so any `ulimit -v` low
    # enough to be a useful cap kills the shell before main().
    DEFAULT_MEM_LIMIT_MB = 4096

    def __init__(self, config):
        super().__init__(config)
        exec_cfg = config.get("execution", {})
        self.mem_limit_mb = int(exec_cfg.get("mem_limit_mb",
                                             self.DEFAULT_MEM_LIMIT_MB))
        self.js = os.path.join(self.ffl_root, "projects", "spidermonkey",
                               "sm-src", "firefox", "js", "src", "obj",
                               "dist", "bin", "js")

    # -- flag selection ----------------------------------------------------

    def _choose_flags(self, facts):
        """Assemble one execution's flag set.

        Draws one group at a time rather than sampling a flat list, so a
        run reliably gets a tier decision *and* a GC decision instead of
        three GC flags and nothing else.
        """
        flags = list(self.BASE_FLAGS)

        # The seed's own directives: part of what the test exercises.
        flags.extend(f for f in (facts.get("flags") or [])
                     if f not in self.ORACLE_BREAKING_FLAGS)

        # Tier. Mostly upward (a miscompile requires the compiler to have
        # run), sometimes downward to reach the slow paths.
        r = random.random()
        if r < 0.60:
            flags.extend(random.choice(self.TIER_FLAGS).split())
        elif r < 0.78:
            flags.append(random.choice(self.TIER_DOWN_FLAGS))

        if random.random() < 0.35:
            flags.append(random.choice(self.GC_FLAGS))
        if random.random() < 0.40:
            flags.append(random.choice(self.VERIFY_FLAGS))

        # Deduplicate while keeping order: a seed's own directive may
        # repeat one we drew, and contradictory tier flags cancel out.
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

        allocator_may_return_null=1
            a huge allocation returns null instead of aborting with an
            ASan report. Fused scripts ask for absurd array lengths
            constantly, and without this every one is a "crash" that has
            nothing to do with an engine bug.
        """
        asan = ":".join(filter(None, [
            f"hard_rss_limit_mb={self.mem_limit_mb}" if self.mem_limit_mb > 0 else "",
            "allocator_may_return_null=1",
            "symbolize=1",
            "detect_leaks=0",
            "handle_abort=1",
            "handle_segv=1",
            "print_summary=1",
            "exitcode=1",
        ]))
        return (f"ASAN_OPTIONS={asan} "
                f"UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=0")

    def _build_command(self, script, facts):
        flags = " ".join(self._choose_flags(facts))
        module = " --module" if facts.get("is_module") else ""
        return f"{self._sanitizer_env()} {self.js} {flags}{module} {script}"

    # -- execution ---------------------------------------------------------

    def execute(self, seed):
        start = time.time()
        workdir = self._make_workdir()
        cmd = "unknown"
        rc, stdout, stderr = 1, "", ""
        try:
            facts = analyze_seed(seed.content)
            script = os.path.join(workdir, "test.js")
            with open(script, "w", encoding="utf-8") as f:
                f.write(apply_harness(seed.content))
            cmd = self._build_command(script, facts)
            rc, stdout, stderr = self._run_command(cmd, cwd=workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

        output = f"{stdout}\n{stderr}"
        # The seed's directives declare which failures the corpus considers
        # correct for it; classify subtracts those.
        verdict = classify(output, facts)
        result = ExecutionResult(
            return_code=rc,
            stdout=stdout,
            stderr=stderr,
            time=time.time() - start,
            crashed=verdict["is_bug"],
            signature=verdict["signature"],
        )
        result.command = cmd
        return result
