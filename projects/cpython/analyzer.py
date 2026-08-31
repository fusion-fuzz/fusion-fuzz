"""
projects/cpython/analyzer.py — the CPython-specific analysis half of the
adapter.

CPython differs from every other target in this repo in one way that
shapes the whole oracle: the seed is *executed*, not just compiled. So a
Python-level exception is the expected outcome of fusing two unrelated
programs — a `TypeError` or `NameError` says the fusion produced nonsense,
not that CPython is broken — and the oracle's main job is to tell the
handful of exceptions that *are* interpreter bugs apart from the flood
that are not.

What counts as a bug
--------------------
1. The interpreter died: `Fatal Python error:`, a signal, an abort.

2. A C-level assertion fired. The build is `--with-pydebug`, so these are
   live. Two formats, neither of which the previous driver's regex
   matched:

       glibc assert():
           file.c:123: func: Assertion `expr' failed.
       CPython's own _PyObject_AssertFailed (Objects/object.c:3232):
           file.c:123: func: Assertion "expr" failed: message

   The old pattern was ``Assertion `expr` failed`` — a backtick on *both*
   sides. glibc closes with a single quote and CPython uses double quotes,
   so it matched neither, and every assertion failure fell through to the
   generic signature.

3. The debug allocator caught corruption — `Debug memory block at address
   p=`, `bad leading pad byte`, `bad ID:` (Objects/obmalloc.c). Only
   present in a pydebug build, and previously not looked for at all.

4. **A C-API contract violation, which arrives as an ordinary-looking
   Python exception.** When a C function returns NULL without setting an
   exception (or returns a result with one set), CPython raises

       SystemError: <built-in ...> returned NULL without setting an exception

   That is a bug in C code — in CPython itself or in an extension — and it
   is indistinguishable from a normal traceback to anything that only
   looks for crashes. This is the class the previous oracle missed
   entirely, and on a fuzzer that executes its input it is the most
   reachable interpreter bug there is.

5. A sanitizer report (the build is `--with-address-sanitizer`).

What does not count
-------------------
Ordinary exceptions, `RecursionError` and `MemoryError` (a fused program
nests arbitrarily deep and allocates arbitrarily much — both are facts
about the input), and the interpreter being killed for memory.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

_RESOURCE_RE = re.compile(
    r"MemoryError"
    r"|RecursionError"
    r"|maximum recursion depth exceeded"
    r"|Cannot allocate memory"
    r"|hard rss limit exhausted"
    r"|allocator is out of memory"
    r"|^Killed$"
    r"|signal: 9",
    re.IGNORECASE | re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s*([^\n]+)")
_FATAL_RE = re.compile(r"Fatal Python error:\s*([^\n]+)")

# Both assertion spellings — see the module docstring for why the previous
# single pattern matched neither.
#   glibc:   file.c:123: func: Assertion `expr' failed.
#   CPython: file.c:123: func: Assertion "expr" failed: message
# Anchored to the start of a line, with an optional `program: ` prefix,
# and the path component bounded.
#
# The unanchored form was catastrophic. `([\w./+-]+\.[ch])` has to try
# every starting offset and every split of a long run of word/dot/slash
# characters before failing, and fused output routinely carries lines tens
# of kilobytes long (a big repr, an -X importtime dump, a giant traceback).
# One `search` over such a line pegged a worker at 100% CPU holding the
# GIL, which starved the other fifteen and wedged the whole campaign —
# twice, for about half an hour each time, while the cumulative throughput
# counter kept reporting 14.7 tests/s. Both real formats begin a line:
#     python: Objects/dictobject.c:1234: insertdict: Assertion `x' failed.
#     Objects/object.c:275: _Py_NegativeRefcount: Assertion "x" failed: msg
_ASSERT_RE = re.compile(
    r"^(?:[\w./+-]{1,200}: )?([\w./+-]{1,200}\.[ch]):(\d+):[^\n]{0,200}?Assertion\s+"
    r"(?:`([^']*)'|\"([^\"]*)\")\s+failed", re.M)
# The bare form CPython prints when it has no expression to show.
_ASSERT_BARE_RE = re.compile(
    r"^(?:[\w./+-]{1,200}: )?([\w./+-]{1,200}\.[ch]):(\d+):[^\n]{0,200}?"
    r"Assertion failed(?::\s*([^\n]*))?", re.M)

# Objects/obmalloc.c, live only under --with-pydebug.
_DEBUG_ALLOC_RE = re.compile(
    r"(Debug memory block at address|bad leading pad byte|bad trailing pad byte"
    r"|bad ID: Allocated using API|Invalid object pointer)")

# The C-API contract violations — real bugs wearing an exception's clothes.
_C_API_RE = re.compile(
    r"SystemError: ([^\n]*?returned (?:NULL without setting an exception|"
    r"a result with an exception set)[^\n]*)")
# Any other SystemError is also a C-level invariant break, just less
# specific. Kept separate so the specific one wins the signature.
_SYSTEM_ERROR_RE = re.compile(r"^SystemError:\s*([^\n]+)", re.M)

# Anchored to the start of a line, and required to stand alone or be
# followed by the shell's "(core dumped)". A bare substring match reported
# `ConnectionAbortedError` — which contains "Aborted", and which appears in
# any program that enumerates the exception hierarchy, inside a *string
# literal* at that. The shell and the runtime always print these on their
# own line.
_SIGNAL_RE = re.compile(
    r"^(Segmentation fault|Bus error|Aborted|Floating point exception|"
    r"Illegal instruction)(?:\s*\(core dumped\))?\s*$", re.M)

# The last frame of a Python traceback, used to group ordinary exceptions
# when one turns out to matter (SystemError).
_TRACEBACK_LINE_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (\S+)', re.M)

_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"\bat \d+\b"), "at N"),
    (re.compile(r"\b\d{5,}\b"), "N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def is_resource_exhaustion(output):
    """True when the interpreter ran out of memory, stack or time.

    RecursionError and MemoryError count: a fused program nests and
    allocates arbitrarily, so both are properties of the input rather than
    defects. They are also *catchable* Python exceptions, which is why they
    have to be excluded explicitly rather than falling out of the
    crash-pattern matching.
    """
    return bool(_RESOURCE_RE.search(output or ""))


# A diagnostic quoted inside a traceback is text the program *printed*,
# not something the interpreter did. CPython's own tests assert on the
# exact wording of allocator diagnostics, so their assertion messages
# contain "Debug memory block at address", "bad ID: Allocated using API"
# and friends verbatim. Matching those reports the test's expectation as
# a finding.
_QUOTED_DIAGNOSTIC_RE = re.compile(
    r"^(?:\s*)(?:AssertionError|self\.assert\w+|Regex didn't match)",
    re.M)


def _diagnostic_is_quoted(text, match_start):
    """True when the matched text sits inside a Python-level assertion.

    The test is deliberately coarse: an allocator diagnostic the
    interpreter really emitted appears on its own line at column zero,
    while a quoted one is embedded in an AssertionError's message.
    """
    line_start = text.rfind("\n", 0, match_start) + 1
    line = text[line_start:text.find("\n", match_start)]
    return bool(_QUOTED_DIAGNOSTIC_RE.match(line)) or line[:1].isspace() \
        or "Regex didn't match" in line or 'assert' in line.lower()


def classify(output):
    """Categorise the interpreter's combined stdout+stderr.

    Returns a dict with:
        kind       "sanitizer", "ubsan", "fatal", "assert", "memory",
                   "c-api", "signal", "resource", "exception", "clean"
        signature  stable grouping key, or None when not a finding
        is_bug     whether this should be saved

    Order matters twice over. A sanitizer report is more precise than the
    abort that follows it, so it comes first; and resource exhaustion has
    to be recognised before the generic exception handling, because
    MemoryError and RecursionError are ordinary exceptions that mean
    nothing about CPython.
    """
    text = output or ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "is_bug": True}

    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {_normalize(m.group(1))}")

    # Resource exhaustion *before* the fatal-error handling, not after.
    # Hitting the driver's own `hard_rss_limit_mb` makes ASan abort, and
    # CPython's fatal handler then prints "Fatal Python error: Aborted" —
    # so checking the fatal pattern first files our own memory cap as an
    # interpreter bug. One of the first five findings was exactly that.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "is_bug": False}

    m = _FATAL_RE.search(text)
    if m:
        return hit("fatal", f"Fatal Python error: {_normalize(m.group(1))}")

    m = _ASSERT_RE.search(text)
    if m:
        expr = m.group(3) if m.group(3) is not None else m.group(4)
        return hit("assert", f"Assertion: {m.group(1)}:{m.group(2)} "
                             f"{_normalize(expr or '')}")
    m = _ASSERT_BARE_RE.search(text)
    if m:
        return hit("assert", f"Assertion: {m.group(1)}:{m.group(2)} "
                             f"{_normalize(m.group(3) or '')}")

    m = _DEBUG_ALLOC_RE.search(text)
    if m and not _diagnostic_is_quoted(text, m.start()):
        return hit("memory", f"Debug allocator: {m.group(1)}")

    # A C function broke its contract with the interpreter. This reads like
    # an ordinary traceback and is a genuine bug — see the module docstring.
    m = _C_API_RE.search(text)
    if m:
        return hit("c-api", f"C-API: {_normalize(m.group(1))}")
    m = _SYSTEM_ERROR_RE.search(text)
    if m:
        return hit("c-api", f"SystemError: {_normalize(m.group(1))}")

    m = _SIGNAL_RE.search(text)
    if m:
        return hit("signal", m.group(1))

    # An ordinary exception. For a fuzzer that *executes* its input this is
    # the expected outcome of joining two unrelated programs, not a finding.
    if "Traceback (most recent call last)" in text or re.search(
            r"^\w*(?:Error|Exception|Warning):", text, re.M):
        return {"kind": "exception", "signature": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# Constructs that make a seed unsafe or useless to run in a fuzzing loop.
# Unlike the compile-only adapters, every seed here actually executes, so
# these are containment concerns rather than validity ones.
_NETWORK_RE = re.compile(
    r'\b(?:socket\.(?:socket|bind|connect|create_server)|'
    r'http\.server|socketserver|urllib\.request\.urlopen|'
    r'asyncio\.(?:start_server|open_connection))\b')
_SUBPROCESS_RE = re.compile(
    r'\b(?:subprocess\.(?:run|Popen|call|check_output)|os\.(?:system|exec\w+|fork|spawn\w+)|'
    r'multiprocessing\.(?:Process|Pool))\b')
_FS_WRITE_RE = re.compile(
    r'\b(?:shutil\.rmtree|os\.(?:remove|unlink|rmdir|removedirs)|'
    r'open\s*\([^)]*[\'"][wax])\b')
_BLOCKING_RE = re.compile(r'\b(?:input\s*\(|sys\.stdin\.read|time\.sleep\s*\(\s*[6-9]\d)')

# unittest/pytest scaffolding: the file defines tests but running it as a
# script executes nothing, so the child is a no-op however it is fused.
_TEST_ONLY_RE = re.compile(r'^\s*(?:import unittest|from unittest)', re.M)
_HAS_MAIN_RE = re.compile(r'^if __name__\s*==', re.M)

_CTYPES_RE = re.compile(r'\b(?:import ctypes|from ctypes)\b')
# Direct code-object manipulation crashes the interpreter by design; a hit
# says nothing about CPython being wrong.
_CODE_OBJECT_RE = re.compile(r'\b__code__\b|\btypes\.CodeType\b|\bcompile\s*\(')


def analyze_seed(content, filename=""):
    """Facts about a Python seed the driver and corpus filter both need.

    Returns a dict:
        uses_network / uses_subprocess / writes_fs / blocks
            Containment concerns. CPython is the one target here whose
            seeds run, so a seed that binds a port or forks is a problem
            for a 16-way concurrent loop in a way that no compile-only
            seed is.
        test_only    defines unittest cases but never runs them
        uses_ctypes  ctypes can segfault the interpreter from pure Python
                     by design, so a crash from one is not a CPython bug
        touches_code_objects
                     same argument for hand-built code objects
        is_runnable  whether executing the file does anything at all
    """
    text = content or ""
    return {
        "uses_network": bool(_NETWORK_RE.search(text)),
        "uses_subprocess": bool(_SUBPROCESS_RE.search(text)),
        "writes_fs": bool(_FS_WRITE_RE.search(text)),
        "blocks": bool(_BLOCKING_RE.search(text)),
        "test_only": bool(_TEST_ONLY_RE.search(text)) and not bool(_HAS_MAIN_RE.search(text)),
        "uses_ctypes": bool(_CTYPES_RE.search(text)),
        "touches_code_objects": bool(_CODE_OBJECT_RE.search(text)),
        "is_runnable": bool(text.strip()),
    }
