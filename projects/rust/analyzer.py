"""
projects/rust/analyzer.py — the Rust-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/rust/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what rustc printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a .rs file that
    determine how it must be compiled and how it may be fused.

They live apart from driver.py because they are pure text functions: no
compiler, no temp directory, no subprocess, so they can be tested
directly.

What a rustc failure looks like
-------------------------------
Four shapes, and they carry very different amounts of information:

    error: internal compiler error: <message>
        A deliberate `bug!`/`span_bug!`. The message is the key.

    thread 'rustc' panicked at compiler/rustc_middle/src/ty/layout.rs:812:9:
    assertion failed: ...
        A `debug_assert!` firing — only visible because setup.py builds
        with rust.debug-assertions. The file:line is the key, and it is a
        better one than the message, which often embeds the offending type.

    query stack during panic:
    #0 [layout_of] computing layout of `Foo`
        Rust's incremental query graph unwinding. The *query name* is the
        single most useful grouping key rustc produces, and no other
        compiler in this repo has an equivalent.

    LLVM ERROR: / Assertion `...' failed.
        The LLVM backend. Only reachable because setup.py enables
        llvm.assertions — bootstrap.example.toml notes that without them
        rustc/LLVM integration bugs "can lead to unsoundness (segfaults,
        etc.) in the rustc process itself".

`error: the compiler unexpectedly panicked. This is a bug` accompanies
the second and third but carries no detail of its own, so it is used only
as a fallback signal.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# The machine ran out of something. Not bugs — the same input compiles on
# a bigger box, so there is nothing to reduce or file.
_RESOURCE_RE = re.compile(
    r"memory allocation of \d+ bytes failed"
    r"|Cannot allocate memory"
    r"|out of memory"
    r"|rustc exited with signal: 9"          # OOM killer
    r"|signal: 9, SIGKILL"
    r"|No space left on device"
    r"|^Killed$",
    re.IGNORECASE | re.M)

# The compiler ran out of *stack*, which for fusion output usually means
# the input nested deeply enough to exhaust it rather than that rustc has a
# defect. Recursion limits are a property of the input's depth.
_STACK_RE = re.compile(r"thread '.*' has overflowed its stack"
                       r"|reached the recursion limit"
                       r"|reached the type-length limit"
                       r"|query depth increased")

_ICE_RE = re.compile(r"internal compiler error(?:\[E\d+\])?:\s*([^\n]+)")
# "thread 'rustc' (9270) panicked at compiler/rustc_middle/src/ty/layout.rs:812:9:"
#
# The `(9270)` is the thread id, which current rustc prints and older
# versions did not. Without allowing for it the location is lost and every
# debug-assertion failure collapses to "(no location)" — which is the best
# grouping key this oracle has, so the whole panic bucket would merge.
_PANIC_AT_RE = re.compile(
    r"thread '[^']*'(?:\s*\(\d+\))? panicked at ([\w/.-]+\.rs):(\d+):\d+:"
    r"\s*\n?\s*([^\n]*)")
# "#0 [layout_of] computing layout of `Foo`"
_QUERY_RE = re.compile(r"^#0 \[(\w+)\]", re.M)
_LLVM_ERROR_RE = re.compile(r"LLVM ERROR:\s*([^\n]+)")
# LLVM's own assertion, reachable because llvm.assertions is on.
_LLVM_ASSERT_RE = re.compile(
    r"Assertion `([^']*)' failed\.|(\S+\.(?:cpp|h)):(\d+): .*Assertion")
_UNEXPECTED_PANIC_RE = re.compile(r"the compiler unexpectedly panicked")
_SIGNAL_RE = re.compile(r"rustc interrupted by (SIG\w+)|signal: \d+, (SIG\w+)")

# Sanitizer and Miri findings in a *compiled program* — the unsafe-Rust
# oracles. A plain panic from the program is not one of these: it is the
# program doing what it was written to do.
_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_MIRI_RE = re.compile(r"error: Undefined Behavior:\s*([^\n]+)")

# Backtrace frames: "  13: rustc_middle::ty::layout::layout_of"
_FRAME_RE = re.compile(r"^\s*\d+:\s+((?:rustc|core|alloc|std)[\w:<>, ]*)", re.M)
_REPORTING_FRAMES = (
    "rustc_errors::", "rustc_driver_impl::", "std::panicking",
    "core::panicking", "rust_begin_unwind", "rustc_middle::util::bug",
    "rustc_middle::ty::context::tls",
)

# Detail that varies run to run and would defeat deduplication.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"\bDefId\([^)]*\)"), "DefId(..)"),
    (re.compile(r"\b_\d+\b"), "_N"),            # MIR locals
    (re.compile(r"\bbb\d+\b"), "bbN"),          # MIR basic blocks
    (re.compile(r"#\d+"), "#N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def is_resource_exhaustion(output):
    """True when the compiler died for lack of memory, stack, disk or time.

    Stack overflow and the recursion/type-length limits count here: fusion
    produces deeply nested types as a matter of course, and rustc running
    out of stack on one is a fact about the input's depth.
    """
    text = output or ""
    return bool(_RESOURCE_RE.search(text) or _STACK_RE.search(text))


def _failing_frame(output):
    for frame in _FRAME_RE.findall(output or ""):
        frame = frame.strip()
        if not any(frame.startswith(r) for r in _REPORTING_FRAMES):
            return frame
    return None


def classify(output):
    """Categorise rustc's combined stdout+stderr.

    Returns a dict with:
        kind       "llvm", "ub", "sanitizer", "ice", "panic", "signal",
                   "resource", "diagnostic", "clean"
        signature  stable grouping key, or None when not a finding
        query      the rustc query that was running, when one was
        is_bug     whether this should be saved

    Order matters: resource exhaustion has to be recognised before the
    panic handling, because rustc reports running out of memory through
    the same panic machinery as a real bug.
    """
    text = output or ""
    query_m = _QUERY_RE.search(text)
    query = query_m.group(1) if query_m else None
    suffix = f" [{query}]" if query else ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "query": query,
                "is_bug": True}

    # Sanitizer / Miri findings in a compiled program. These are the
    # unsafe-Rust oracles and are more precise than anything that follows.
    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _MIRI_RE.search(text)
    if m:
        return hit("ub", f"UB: {_normalize(m.group(1))}")

    # LLVM's own checks, live because setup.py enables llvm.assertions.
    m = _LLVM_ERROR_RE.search(text)
    if m:
        return hit("llvm", f"LLVM ERROR: {_normalize(m.group(1))}{suffix}")
    m = _LLVM_ASSERT_RE.search(text)
    if m:
        detail = m.group(1) or f"{m.group(2)}:{m.group(3)}"
        return hit("llvm", f"LLVM assert: {_normalize(detail)}{suffix}")

    # Before any panic handling.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "query": query,
                "is_bug": False}

    # A debug_assert! firing. The source location groups better than the
    # message, which usually embeds the offending type.
    m = _PANIC_AT_RE.search(text)
    if m and "compiler/" in m.group(1):
        # Location and query only — deliberately not the panic message.
        #
        # The message routinely embeds the offending type ("Binder { value:
        # WellFormed(Term::Ty([(); ...") which differs for every seed that
        # reaches the same assertion, so including it files one bug per
        # seed instead of one per bug. A source line holds one assertion,
        # so the location is both stable and specific. The message is still
        # in the saved bundle's test.out for triage.
        where = f"{m.group(1)}:{m.group(2)}"
        return hit("panic", f"PANIC: {where}{suffix}")

    m = _ICE_RE.search(text)
    if m:
        return hit("ice", f"ICE: {_normalize(m.group(1))}{suffix}")

    m = _SIGNAL_RE.search(text)
    if m:
        sig = m.group(1) or m.group(2)
        frame = _failing_frame(text)
        return hit("signal", f"{sig}{suffix}" + (f" in {frame}" if frame else ""))

    if _UNEXPECTED_PANIC_RE.search(text):
        frame = _failing_frame(text)
        return hit("panic", f"PANIC: (no location){suffix}"
                            + (f" in {frame}" if frame else ""))

    # Ordinary diagnostics — "error[E0308]: mismatched types". The expected
    # outcome for most fused programs.
    if re.search(r"^error(\[E\d+\])?:", text, re.M) or "warning:" in text:
        return {"kind": "diagnostic", "signature": None, "query": query,
                "is_bug": False}
    return {"kind": "clean", "signature": None, "query": query, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# compiletest headers. `//@ compile-flags: ...` carries the flags the test
# was written to exercise, the way GCC's dg-options does.
_COMPILE_FLAGS_RE = re.compile(r"^//@\s*compile-flags:\s*(.+)$", re.M)
_EDITION_RE = re.compile(r"^//@\s*edition:\s*(\d+)$", re.M)
_CHECK_PASS_RE = re.compile(r"^//@\s*(check-pass|build-pass|run-pass|check-fail|"
                            r"build-fail|run-fail|known-bug)\b", re.M)

# Flags safe to pass through: no paths, no target changes (the driver picks
# the target itself), nothing that needs an auxiliary crate.
_SAFE_FLAG_RE = re.compile(
    r'^-(?:C\s*[\w=-]+|Z\s*[\w=-]+|O|g|W\s*[\w-]+|A\s*[\w-]+|D\s*[\w-]+)$'
    r'|^--edition=\d+$')

_UNSAFE_BLOCK_RE = re.compile(r'\bunsafe\s*(?:\{|fn\b|impl\b|trait\b)')
_RAW_PTR_RE = re.compile(r'\*(?:const|mut)\s+\w')
_TRANSMUTE_RE = re.compile(r'\btransmute\b|\bfrom_raw_parts\b|\bMaybeUninit\b'
                           r'|\bzeroed\b|\bassume_init\b|\bunion\b')
_NO_STD_RE = re.compile(r'^\s*#!\[no_std\]', re.M)
_NO_MAIN_RE = re.compile(r'^\s*#!\[no_main\]', re.M)
_MAIN_RE = re.compile(r'^\s*(?:pub\s+)?fn\s+main\s*\(', re.M)
_FEATURE_RE = re.compile(r'^\s*#!\[feature\(([^)]*)\)\]', re.M)
_CRATE_ATTR_RE = re.compile(r'^\s*#!\[[^\]]*\]', re.M)


def compile_flags_of(content):
    """The safe subset of a test's own `//@ compile-flags:`.

    They are part of what the test exercises — a mir-opt test compiled
    without `-Z mir-opt-level=4` exercises nothing in particular — but
    flags naming paths or targets would fail in the driver rather than in
    the compiler, so those are dropped.
    """
    m = _COMPILE_FLAGS_RE.search(content or "")
    if not m:
        return []
    return [f for f in m.group(1).split() if _SAFE_FLAG_RE.match(f)][:6]


def analyze_seed(content, filename=""):
    """Facts about a Rust seed that fusion and execution both need.

    Returns a dict:
        edition        the test's declared edition, if any
        expectation    check-pass / known-bug / ... when declared
        compile_flags  the safe subset of its own compile-flags
        features       #![feature(...)] gates, which must be hoisted to the
                       crate root when two files are fused
        crate_attrs    every inner attribute, for the same reason
        no_std/no_main whether the file opts out of the prelude or of the
                       generated main
        has_main       whether it defines `fn main`
        unsafe_score   a count of unsafe-Rust constructs — raw pointers,
                       transmute, unions, MaybeUninit. The driver uses it
                       to decide when the sanitizers are worth their cost.
        is_known_bug   `//@ known-bug` marks a test that is *supposed* to
                       ICE. Fusing it and reporting the ICE would be
                       rediscovering a filed bug.
    """
    text = content or ""
    m = _EDITION_RE.search(text)
    edition = m.group(1) if m else None
    m = _CHECK_PASS_RE.search(text)
    expectation = m.group(1) if m else None

    features = []
    for m in _FEATURE_RE.finditer(text):
        features.extend(f.strip() for f in m.group(1).split(",") if f.strip())

    unsafe_score = (len(_UNSAFE_BLOCK_RE.findall(text))
                    + len(_RAW_PTR_RE.findall(text))
                    + len(_TRANSMUTE_RE.findall(text)))

    return {
        "edition": edition,
        "expectation": expectation,
        "compile_flags": compile_flags_of(text),
        "features": sorted(set(features)),
        "crate_attrs": _CRATE_ATTR_RE.findall(text),
        "no_std": bool(_NO_STD_RE.search(text)),
        "no_main": bool(_NO_MAIN_RE.search(text)),
        "has_main": bool(_MAIN_RE.search(text)),
        "unsafe_score": unsafe_score,
        "is_known_bug": expectation == "known-bug",
    }
