"""
projects/spidermonkey/analyzer.py — the SpiderMonkey-specific analysis
half of the adapter.

Two kinds of analysis, both consumed by projects/spidermonkey/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what the shell printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a .js file that
    determine how it must be run.

Like V8 and CPython, this is an *executing* target: a fused program throws
`TypeError` far more often than not, and that says the fusion produced
nonsense, not that the engine is broken. The oracle's real work is
separating the few outputs that are engine bugs from that flood.

What a SpiderMonkey failure looks like
--------------------------------------
Both shapes are taken from mfbt/Assertions.h rather than from memory:

    [1234] Assertion failure: expr, at /path/js/src/jit/Foo.cpp:567
        MOZ_ASSERT. This is SpiderMonkey's assertion mechanism and the
        direct counterpart of V8's DCHECK; it exists only because
        setup.py configures with --enable-debug, and nearly every
        interesting finding arrives through it.

    [1234] Hit MOZ_CRASH(reason) at /path/js/src/jit/Foo.cpp:567
        MOZ_CRASH — a deliberate, unconditional abort on a state the
        engine considers impossible.

    ==1234==ERROR: AddressSanitizer: heap-use-after-free ...
        The build is ASan. A JIT type confusion usually surfaces here.

The thing this engine has that V8 does not
------------------------------------------
Its corpus declares expected failures. A jit-test file can carry

    // |jit-test| allow-oom; error: TypeError; exitstatus: 3

which says the test is *supposed* to run out of memory, or to throw, or to
exit non-zero. Those directives travel with the seed into the fused
program, and honouring them is what keeps a deliberate failure from being
filed as a finding — the same class of mistake as V8's
`triggerAssertFalse`.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that mean the machine (or our own cap) ran out of something.
# Checked before the assertion handling: SpiderMonkey reports OOM through
# its own MOZ_CRASH in places, so "Hit MOZ_CRASH" alone would file every
# out-of-memory as an engine bug.
_RESOURCE_RE = re.compile(
    r"out of memory"
    r"|OOM in |MOZ_CRASH\(Out of memory\)"
    r"|Hit MOZ_CRASH\(out of memory"
    r"|allocation size overflow"
    r"|too much recursion"
    r"|InternalError: too much recursion"
    r"|hard rss limit exhausted"
    # The shell's own watchdog, which the execution harness arms around
    # its heat loop (projects/spidermonkey/driver.py). It means the
    # program ran too long, not that the engine is broken.
    r"|Script terminated by interrupt handler"
    r"|^Killed$",
    re.IGNORECASE | re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s*([^\n]+)")

# mfbt/Assertions.h:
#   fprintf(stderr, "[%d] Assertion failure: %s, at %s:%d\n", ...)
#   fprintf(stderr, "[%d] Hit MOZ_CRASH(%s) at %s:%d\n", ...)
_ASSERT_RE = re.compile(
    r"^\[\d+\] Assertion failure: (.*?), at ([^\s:]+):(\d+)\s*$", re.M)
_MOZ_CRASH_RE = re.compile(
    r"^\[\d+\] Hit MOZ_CRASH\((.*?)\) at ([^\s:]+):(\d+)\s*$", re.M)

# A MOZ_CRASH raised while the shell parses its own command line, e.g.
# "invalid option value for --nursery-strings, must be on/off". The engine
# never started; the command line was wrong.
_SHELL_OPTION_RE = re.compile(r"invalid option|unknown option|option value for")

_SIGNAL_RE = re.compile(
    r"^(Segmentation fault|Bus error|Aborted|Illegal instruction|"
    r"Floating point exception)(?:\s*\(core dumped\))?\s*$", re.M)

# An uncaught JavaScript exception. For a fuzzer that executes its input
# this is the expected outcome of joining two unrelated programs.
_JS_EXCEPTION_RE = re.compile(
    r"^\S*:\d+:\d*\s*(TypeError|ReferenceError|SyntaxError|RangeError|"
    r"URIError|EvalError|AggregateError|InternalError|Error):", re.M)
_THROWN_RE = re.compile(
    r"^(?:uncaught exception: )?(\w*Error)(?::|\b)", re.M)

# Detail that varies run to run and would defeat deduplication.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"^.*?/js/src/"), "js/src/"),
    (re.compile(r"\b\d{6,}\b"), "N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def _short_path(path):
    """Keep the part of a source path that identifies the file.

    A debug build embeds absolute paths, and the build directory differs
    between machines; only the js/src-relative tail is stable.
    """
    path = path.replace("\\", "/")
    idx = path.find("js/src/")
    return path[idx:] if idx >= 0 else path.rsplit("/", 1)[-1]


def is_resource_exhaustion(output):
    """True when the shell died for lack of memory or stack.

    `too much recursion` counts: fusion produces deep recursion as a
    matter of course, and SpiderMonkey reports it as a catchable
    InternalError, so it cannot be excluded by the crash patterns alone.
    """
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output, expectations=None):
    """Categorise the shell's combined stdout+stderr.

    Returns a dict with:
        kind       "sanitizer", "ubsan", "assert", "moz_crash", "signal",
                   "resource", "expected", "exception", "clean"
        signature  stable grouping key, or None when not a finding
        is_bug     whether this should be saved

    *expectations* is analyze_seed()'s output for the program that was
    run. It carries the jit-test directives, which declare failures the
    corpus considers correct — an `allow-oom` test is *supposed* to run
    out of memory.

    Order is load-bearing twice: the sanitizer report is more precise than
    the abort that follows it, so it comes first; and resource exhaustion
    must be recognised before the MOZ_CRASH handling, because
    SpiderMonkey reports some OOM conditions through MOZ_CRASH.
    """
    text = output or ""
    exp = expectations or {}

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "is_bug": True}

    # Resource exhaustion first: SpiderMonkey reports some out-of-memory
    # conditions through MOZ_CRASH, the same channel a genuine
    # impossible-state abort uses.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "is_bug": False}

    # A crash while the shell is parsing its own arguments is a bad
    # command line, not an engine bug. It reproduces on an empty script,
    # and it would otherwise be reported once per execution.
    m = _MOZ_CRASH_RE.search(text)
    if m and _SHELL_OPTION_RE.search(m.group(1)):
        return {"kind": "shell_option", "signature": None, "is_bug": False}

    # MOZ_ASSERT and MOZ_CRASH are checked *before* the sanitizer report,
    # unlike every other adapter here. Under an ASan build the deliberate
    # abort those macros perform is itself intercepted, and ASan then
    # reports `SEGV ... Assertions.h:261 in MOZ_CrashSequence` — the same
    # text for every assertion in the engine. Taking that as the signature
    # would collapse every distinct assertion into one group and hide
    # where the failure actually was.
    m = _ASSERT_RE.search(text)
    if m:
        where = f"{_short_path(m.group(2))}:{m.group(3)}"
        return hit("assert", f"ASSERT: {where}")

    m = _MOZ_CRASH_RE.search(text)
    if m:
        reason = _normalize(m.group(1))
        where = f"{_short_path(m.group(2))}:{m.group(3)}"
        # A seed that declares it may hit an unhandlable OOM is describing
        # this exact crash as correct behaviour.
        if exp.get("allow_unhandlable_oom") and "memory" in reason.lower():
            return {"kind": "expected", "signature": None, "is_bug": False}
        return hit("moz_crash", f"MOZ_CRASH: {where} ({reason})")

    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {_normalize(m.group(1))}")

    m = _SIGNAL_RE.search(text)
    if m:
        return hit("signal", m.group(1))

    if _JS_EXCEPTION_RE.search(text) or _THROWN_RE.search(text):
        return {"kind": "exception", "signature": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "is_bug": False}


def crash_signature(output, expectations=None):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output, expectations)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# jit-test declares how a test must be run, and what failing means, in a
# leading comment:
#   // |jit-test| --ion-eager; allow-oom; error: TypeError
_DIRECTIVE_RE = re.compile(r"^//\s*\|jit-test\|\s*(.+)$", re.M)

# Flags safe to pass through. Anything naming a path or a file would refer
# to the test's original directory, which the seed has left.
_SAFE_FLAG_RE = re.compile(r'^--[\w-]+(?:=[\w,.:-]+)?$')

# Ceiling on a numeric flag value. The engine sizes internal limits
# directly from some of these, so an absurd one aborts during startup —
# a crash that reproduces on an empty script and says nothing about how
# the engine handles JavaScript.
_MAX_FLAG_VALUE = 10 ** 7

# `load()` and `evaluate(read(...))` pull in a sibling script — usually
# one of jit-test's own lib/ helpers. A fused seed no longer sits next to
# it.
_LOAD_RE = re.compile(r'\b(?:load|loadRelativeToScript|snarf|read)\s*\(')

# Shell-only testing functions. They are the SpiderMonkey analogue of V8's
# %-natives: the engine's internals exposed directly to script. Unlike
# V8's, they are ordinary identifiers, so a seed using one is still
# *parseable* without a flag — it just throws ReferenceError.
_SHELL_FN_RE = re.compile(
    r'\b(?:gc|gczeal|gcparam|oomTest|oomAtAllocation|stackTest|'
    r'inIon|inJit|assertJitStackInvariants|getBuildConfiguration|'
    r'evalcx|newGlobal|createIsHTMLDDA|enableTrackAllocations|'
    r'relazifyFunctions|setJitCompilerOption|trialInline|'
    r'wasmIsSupported|readGeckoProfilingStack)\s*\(')

# Constructs that reach the parts of the engine most likely to be wrong:
# the JIT tiers, the GC, and typed-array/ArrayBuffer memory.
_JIT_RE = re.compile(
    r'\b(?:setJitCompilerOption|inIon|inJit|trialInline|'
    r'assertJitStackInvariants|relazifyFunctions)\b')
_MEMORY_RE = re.compile(
    r'\b(?:ArrayBuffer|SharedArrayBuffer|DataView|Atomics|'
    r'(?:Ui|I)nt(?:8|16|32)Array|Float(?:32|64)Array|BigInt64Array|'
    r'WeakRef|FinalizationRegistry)\b')
_WASM_RE = re.compile(r'\bWebAssembly\.|\bwasm(?:TextToBinary|Eval)\s*\(')


def _parse_directives(content):
    """The `// |jit-test|` line, split into flags and expectations.

    The directives are semicolon-separated; a `key: value` form carries an
    argument, a bare word is a switch, and anything starting with `--` is
    a shell flag.
    """
    flags, opts = [], {}
    for m in _DIRECTIVE_RE.finditer(content or ""):
        for part in m.group(1).split(";"):
            part = part.strip()
            if not part:
                continue
            if part.startswith("--"):
                token = part.split()[0]
                if not _SAFE_FLAG_RE.match(token):
                    continue
                _, _, value = token.partition("=")
                if value.isdigit() and int(value) > _MAX_FLAG_VALUE:
                    continue
                if token not in flags:
                    flags.append(token)
            elif ":" in part:
                key, _, value = part.partition(":")
                opts[key.strip().replace("-", "_")] = value.strip()
            else:
                opts[part.replace("-", "_")] = True
    return flags, opts


def analyze_seed(content, filename=""):
    """Facts about a SpiderMonkey seed that the driver and oracle need.

    Returns a dict:
        flags           shell flags from the seed's `// |jit-test|` line
        allow_oom       the test declares that running out of memory is a
                        correct outcome for it
        allow_unhandlable_oom
                        ... including the kind that aborts the process
        expects_error   the test declares it must throw; a failure is the
                        point of it, the way a clang -verify test's
                        diagnostic is
        exit_status     the test declares a non-zero exit code as correct
        is_module       needs --module; its `import` bindings cannot be
                        merged into a fused script the way statements can
        uses_shell_fns  calls the shell's testing functions
        jit_score       how much of the JIT-control surface it pokes
        memory_score    typed arrays, ArrayBuffers, Atomics, weak refs
        uses_wasm       touches WebAssembly
        needs_load      load()s a sibling file that no longer exists
    """
    text = content or ""
    flags, opts = _parse_directives(text)
    return {
        "flags": flags,
        "allow_oom": bool(opts.get("allow_oom")),
        "allow_unhandlable_oom": bool(opts.get("allow_unhandlable_oom")),
        "expects_error": opts.get("error") or None,
        "exit_status": opts.get("exitstatus") or None,
        "is_module": bool(opts.get("module")),
        "uses_shell_fns": bool(_SHELL_FN_RE.search(text)),
        "jit_score": len(_JIT_RE.findall(text)),
        "memory_score": len(_MEMORY_RE.findall(text)),
        "uses_wasm": bool(_WASM_RE.search(text)),
        "needs_load": bool(_LOAD_RE.search(text)),
    }
