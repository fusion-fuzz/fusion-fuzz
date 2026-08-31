"""
projects/v8/analyzer.py — the V8-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/v8/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what d8 printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a .js file that
    determine how it must be run.

V8 is an *executing* target, like CPython and unlike the compilers: a
fused program throws `TypeError` or `ReferenceError` far more often than
not, and that says the fusion produced nonsense, not that V8 is broken.
The oracle's real work is separating the few outputs that are engine bugs
from that flood.

What a V8 failure looks like
----------------------------
All four shapes are taken from the source rather than from memory:

    #
    # Fatal error in ../../src/objects/foo.cc, line 123
    # Debug check failed: expr.
        A DCHECK. `V8_Fatal` prints exactly this header
        (src/base/logging.cc), and the "Debug check failed: %s." body comes
        from `DCheckHelper`. Only present because setup.py builds with
        is_debug and v8_enable_slow_dchecks — this is V8's assertion
        mechanism and the main thing the debug build buys.

    ## V8 sandbox violation detected!
        The sandbox caught a write (or read) outside the heap cage
        (src/sandbox/). On a JIT engine this is the highest-signal report
        there is: it means a memory-safety invariant broke, not merely
        that something crashed.

    ==1234==ERROR: AddressSanitizer: heap-use-after-free ...
        The build is is_asan. A JIT type confusion usually surfaces here.

    #
    # Fatal javascript OOM in ...
        Not a bug. V8 reports running out of memory through the same
        `Fatal error` channel as a real DCHECK, so the resource shapes have
        to be subtracted explicitly and *before* the fatal handling.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that mean the machine (or our own cap) ran out of something.
# Checked before the fatal handling: V8 routes OOM through V8_Fatal, so
# "Fatal error" alone would file every OOM as a DCHECK failure.
_RESOURCE_RE = re.compile(
    r"Fatal javascript OOM"
    r"|Fatal process out of memory"
    r"|FatalProcessOutOfMemory"
    r"|Reached heap limit"
    r"|JavaScript heap out of memory"
    r"|Allocation failed - (?:process|JavaScript) (?:out of memory|heap)"
    r"|RangeError: Maximum call stack size exceeded"
    r"|hard rss limit exhausted"
    # V8 turns a plain JS stack overflow into a FATAL under
    # --correctness-fuzzer-suppressions (src/execution/isolate.cc). The
    # driver no longer passes that flag, but a seed's own `// Flags:` line
    # can still carry it, so the shape is subtracted here too.
    r"|Aborting on stack overflow"
    r"|out of memory"
    r"|^Killed$",
    re.IGNORECASE | re.M)

_SANDBOX_RE = re.compile(r"V8 sandbox violation detected")

# Aborts the *test* asked for, not failures of the engine. The driver
# already passes --disable-abortjs and withholds --expose-trigger-failure,
# but a seed reaching these another way must not be filed as a finding.
_DELIBERATE_RE = re.compile(
    r"^abort: "                                    # %AbortJS
    r"|trigger-failure-extension\.cc"              # trigger{Assert,Check}False
    r"|Check failed: false\.",                     # CHECK(false) verbatim
    re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s*([^\n]+)")

# src/base/logging.cc:
#   PrintError("\n\n#\n# Fatal error in %s, line %d\n# ", file, line)
#   PrintError("\n\n#\n# Fatal error\n# ")
# followed by the message, which for a DCHECK is "Debug check failed: %s."
_FATAL_AT_RE = re.compile(
    r"#\s*Fatal error in ([^,\n]+), line (\d+)\s*\n#\s*([^\n]*)")
_FATAL_RE = re.compile(r"#\s*Fatal error\s*\n#\s*([^\n]*)")
_DCHECK_RE = re.compile(r"(?:Debug c|C)heck failed:\s*([^\n]+)")

# V8 prints its own stack after a fatal; the first frame inside v8::
# internal is the useful grouping key.
_FRAME_RE = re.compile(r"^\s*#?\d+\s+[\dxa-f]*\s*(v8::internal::[\w:<>~]+)", re.M)
_REPORTING_FRAMES = (
    "v8::internal::V8_Fatal", "v8::internal::FatalError",
    "v8::base::OS::Abort", "v8::internal::Isolate::Throw",
)

_SIGNAL_RE = re.compile(
    r"^(Segmentation fault|Bus error|Aborted|Illegal instruction)"
    r"(?:\s*\(core dumped\))?\s*$", re.M)

# An uncaught JavaScript exception. For a fuzzer that executes its input
# this is the expected outcome of joining two unrelated programs.
_JS_EXCEPTION_RE = re.compile(
    r"^\S*:\d+:.*?\b(TypeError|ReferenceError|SyntaxError|RangeError|"
    r"URIError|EvalError|AggregateError|Error):", re.M)
_THROWN_RE = re.compile(r"^(?:Uncaught )?(\w*Error)(?::|\b)", re.M)

# Detail that varies run to run and would defeat deduplication.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"\.\./\.\./"), ""),
    (re.compile(r"\b\d{6,}\b"), "N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def is_resource_exhaustion(output):
    """True when d8 died for lack of memory or stack.

    `RangeError: Maximum call stack size exceeded` counts: fusion produces
    deep recursion as a matter of course, and it is a catchable JavaScript
    exception, so it cannot be excluded by the crash patterns alone.
    """
    return bool(_RESOURCE_RE.search(output or ""))


def _failing_frame(output):
    for frame in _FRAME_RE.findall(output or ""):
        if not any(frame.startswith(r) for r in _REPORTING_FRAMES):
            return frame
    return None


def classify(output):
    """Categorise d8's combined stdout+stderr.

    Returns a dict with:
        kind       "sandbox", "sanitizer", "ubsan", "dcheck", "fatal",
                   "signal", "resource", "exception", "clean"
        signature  stable grouping key, or None when not a finding
        is_bug     whether this should be saved

    Order is load-bearing twice: the sandbox report and the sanitizer
    report are more precise than the abort that follows them, so they come
    first; and resource exhaustion must be recognised before the fatal
    handling, because V8 reports OOM through the same `Fatal error`
    channel as a genuine DCHECK failure.
    """
    text = output or ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "is_bug": True}

    # The sandbox catching an out-of-cage access is the strongest signal
    # V8 produces — a memory-safety invariant broke.
    if _SANDBOX_RE.search(text):
        frame = _failing_frame(text)
        return hit("sandbox",
                   f"SANDBOX: violation{f' in {frame}' if frame else ''}")

    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {_normalize(m.group(1))}")

    # Before any `Fatal error` handling — see the docstring.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "is_bug": False}

    # Also before it: a crash the input deliberately asked for is not a
    # finding, and it arrives wearing the same DCHECK/abort clothes as a
    # real one.
    if _DELIBERATE_RE.search(text):
        return {"kind": "deliberate", "signature": None, "is_bug": False}

    # A DCHECK. The file:line groups better than the expression, which
    # often embeds the offending value.
    m = _FATAL_AT_RE.search(text)
    if m:
        where = f"{_normalize(m.group(1))}:{m.group(2)}"
        detail = m.group(3)
        d = _DCHECK_RE.search(detail) or _DCHECK_RE.search(text)
        kind = "dcheck" if d else "fatal"
        return hit(kind, f"{'DCHECK' if d else 'FATAL'}: {where}")

    m = _DCHECK_RE.search(text)
    if m:
        return hit("dcheck", f"DCHECK: {_normalize(m.group(1))}")
    m = _FATAL_RE.search(text)
    if m:
        return hit("fatal", f"FATAL: {_normalize(m.group(1))}")

    m = _SIGNAL_RE.search(text)
    if m:
        frame = _failing_frame(text)
        return hit("signal", f"{m.group(1)}{f' in {frame}' if frame else ''}")

    # An uncaught JavaScript exception: the expected outcome of fusing two
    # unrelated programs, and not a finding.
    if _JS_EXCEPTION_RE.search(text) or _THROWN_RE.search(text):
        return {"kind": "exception", "signature": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# mjsunit tests declare what they need in a leading comment:
#   // Flags: --allow-natives-syntax --expose-gc
_FLAGS_RE = re.compile(r"^//\s*Flags:\s*(.+)$", re.M)

# Flags safe to pass through. Anything naming a path or a file would refer
# to the test's original directory, which the seed has left.
_SAFE_FLAG_RE = re.compile(r'^--[\w-]+(?:=[\w,.-]+)?$')

# Ceiling on a numeric flag value. V8 sizes internal limits directly from
# these (`kLimitSize = v8_flags.stack_size * KB`), so an absurd one fails a
# DCHECK during isolate setup — a crash that reproduces on an empty script
# and says nothing about the engine's handling of JavaScript. Real mjsunit
# values are small: --stack-size is 100-1200 across the corpus.
#
# Defence in depth. The mutator no longer rewrites the `// Flags:` line
# (see V8FusionStrategy.fuse), which was where the absurd values came
# from; this catches any that arrive another way.
_MAX_FLAG_VALUE = 10 ** 7

# `load()` and `d8.file.execute()` pull in a sibling script — usually
# mjsunit.js itself. A fused seed no longer sits next to it.
_LOAD_RE = re.compile(r'\b(?:load|d8\.file\.execute)\s*\(')

# %-prefixed runtime functions. These are V8's internal surface — the
# JavaScript equivalent of unsafe code — and only exist under
# --allow-natives-syntax. A seed using one *must* get that flag or it is a
# syntax error, so this is not optional tuning.
_NATIVES_RE = re.compile(r'%[A-Z]\w*\s*\(')

# Constructs that reach the parts of the engine most likely to be wrong:
# the optimising tiers, the GC, and typed-array/ArrayBuffer memory.
_OPTIMIZE_RE = re.compile(
    r'%(?:OptimizeFunctionOnNextCall|PrepareFunctionForOptimization|'
    r'OptimizeMaglevOnNextCall|CompileBaseline|NeverOptimizeFunction|'
    r'DeoptimizeFunction|OptimizeOsr)\b')
_MEMORY_RE = re.compile(
    r'\b(?:ArrayBuffer|SharedArrayBuffer|DataView|Atomics|'
    r'(?:Ui|I)nt(?:8|16|32)Array|Float(?:32|64)Array|BigInt64Array|'
    r'WeakRef|FinalizationRegistry)\b')
_WORKER_RE = re.compile(r'\bnew\s+Worker\s*\(|\bWorker\.')
_WASM_RE = re.compile(r'\bWebAssembly\.')


def flags_of(content):
    """The safe subset of a test's own `// Flags:` line.

    They are part of what the test exercises — an mjsunit test written
    around `%OptimizeFunctionOnNextCall` does nothing without
    --allow-natives-syntax — so they are honoured rather than discarded.
    """
    flags = []
    for m in _FLAGS_RE.finditer(content or ""):
        for f in m.group(1).split():
            if not _SAFE_FLAG_RE.match(f):
                continue
            _, _, value = f.partition("=")
            if value.isdigit() and int(value) > _MAX_FLAG_VALUE:
                continue
            flags.append(f)
    return list(dict.fromkeys(flags))


def analyze_seed(content, filename=""):
    """Facts about a V8 seed that the driver needs.

    Returns a dict:
        flags          the seed's own `// Flags:`, filtered
        uses_natives   contains %-prefixed runtime functions, so it needs
                       --allow-natives-syntax to parse at all
        optimize_score how much of V8's optimising-tier surface it pokes
        memory_score   typed arrays, ArrayBuffers, Atomics, weak refs
        uses_worker    spawns a Worker; d8 supports it, but it makes the
                       run nondeterministic
        uses_wasm      touches WebAssembly
        needs_load     `load()`s a sibling file that no longer exists
    """
    text = content or ""
    return {
        "flags": flags_of(text),
        "uses_natives": bool(_NATIVES_RE.search(text)),
        "optimize_score": len(_OPTIMIZE_RE.findall(text)),
        "memory_score": len(_MEMORY_RE.findall(text)),
        "uses_worker": bool(_WORKER_RE.search(text)),
        "uses_wasm": bool(_WASM_RE.search(text)),
        "needs_load": bool(_LOAD_RE.search(text)),
    }
