"""
projects/tint/analyzer.py — the tint-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/tint/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what tint printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a .wgsl file that
    determine how it must be run.

tint is a *compiler*, not an executor: unlike V8 or CPython, the expected
outcome of a malformed program is a clean diagnostic ("error: ..."), not
an exception. A fused WGSL program is usually ill-formed, so an ordinary
compile error is the common case and not a finding. The oracle's job is to
separate the internal-compiler-errors and sanitizer reports — which are
tint's fault — from that flood of ordinary diagnostics.

What a tint failure looks like
------------------------------
Both internal shapes are taken from src/tint/utils/ice/ice.cc rather than
from memory:

    ../../src/tint/lang/core/ir/foo.cc:123 internal compiler error: TINT_ASSERT(expr)
        A TINT_ASSERT / TINT_UNREACHABLE / TINT_UNIMPLEMENTED. This is
        tint's assertion mechanism (InternalCompilerError::Error()), and
        it is the main thing the assert-enabled build buys. It is followed
        by a `__builtin_trap()`, so the process dies with SIGILL, not
        SIGABRT.

    ==1234==ERROR: AddressSanitizer: heap-use-after-free ...
        The build is ASan/UBSan. A memory-safety bug reached through a
        malformed shader surfaces here.

One ordering note before that: TINT_ICE ends in `__builtin_trap()`, so
under this ASan build ASan intercepts the trap and prints its own report
naming `ice.cc` — identical for every internal error. The ICE line must
therefore be preferred over the sanitizer summary, which is the opposite
of the usual order.

The IR validator deserves its own note: when it rejects tint's own IR it
reports through the same ICE channel, so a validation failure reads as an
`internal compiler error` and is filed as a bug — which is correct, that
is exactly the high-value signal the validator exists to produce.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that mean the machine (or our own cap) ran out of something.
# Checked first: tint can hit these on a pathological fused program, and
# they say nothing about a compiler bug.
_RESOURCE_RE = re.compile(
    r"out of memory"
    r"|std::bad_alloc"
    r"|Allocation of \d+ bytes failed"
    r"|hard rss limit exhausted"
    r"|^Killed$"
    r"|Stack overflow"
    r"|stack overflow",
    re.IGNORECASE | re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s*([^\n]+)")

# src/tint/utils/ice/ice.cc:
#   File() + ":" + Line() + " internal compiler error: " + Message()
# The message for an assertion begins "TINT_ASSERT(expr) ", for an
# unreachable "TINT_UNREACHABLE ", for unimplemented "TINT_UNIMPLEMENTED ".
_ICE_RE = re.compile(
    r"^([^\s:]+\.(?:cc|h|inl)):(\d+)\s+internal compiler error:\s*([^\n]*)",
    re.M)

# The trap TINT_ICE performs (__builtin_trap) surfaces as SIGILL; a real
# memory bug may surface as SIGSEGV before the sanitizer prints. These are
# the bare signal lines a shell prints when tint dies without a report.
_SIGNAL_RE = re.compile(
    r"^(Segmentation fault|Illegal instruction|Aborted|Bus error|"
    r"Trace/breakpoint trap)(?:\s*\(core dumped\))?\s*$", re.M)

# An ordinary WGSL diagnostic — the expected outcome of compiling a fused
# (and usually ill-formed) program. tint prints `<file>:<line>:<col> error: ...`
# and `warning: ...`.
_DIAGNOSTIC_RE = re.compile(r"^[^\n]*?:\d+:\d+\s+error:\s", re.M)

# Detail that varies run to run and would defeat deduplication.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"^.*?/src/tint/"), "src/tint/"),
    (re.compile(r"\.\./\.\./"), ""),
    (re.compile(r"\b\d{6,}\b"), "N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def _short_path(path):
    """Keep the part of a source path that identifies the file.

    A build embeds paths relative to its own out directory (../../src/...);
    only the src/tint-relative tail is stable across machines.
    """
    path = path.replace("\\", "/")
    idx = path.find("src/tint/")
    return path[idx:] if idx >= 0 else path.lstrip("./")


def is_resource_exhaustion(output):
    """True when tint died for lack of memory or stack.

    A fused WGSL program can nest expressions deeply enough to overflow the
    parser's stack; that is a property of the input, not a compiler bug.
    """
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output):
    """Categorise tint's combined stdout+stderr.

    Returns a dict with:
        kind       "sanitizer", "ubsan", "ice", "unimplemented", "signal",
                   "resource", "diagnostic", "clean"
        signature  stable grouping key, or None when not a finding
        is_bug     whether this should be saved

    Order is load-bearing three times:

      * resource exhaustion is subtracted first — tint routes some of it
        through the same abort path as a real failure;
      * the ICE comes before the sanitizer report, because ASan intercepts
        the trap TINT_ICE performs and names ice.cc for every one of them
        (see the module docstring);
      * the ICE also comes before the ordinary-diagnostic check, since an
        ICE line carries a file and line number the diagnostic pattern
        could partially match.
    """
    text = output or ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "is_bug": True}

    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "is_bug": False}

    # The ICE is checked *before* the sanitizer report, unlike most
    # adapters here. TINT_ICE ends in `__builtin_trap()`, and under an ASan
    # build ASan intercepts that trap and reports
    # `ILL ... ice.cc:71 in tint::InternalCompilerError::~InternalCompilerError`
    # — the same text for every ICE in the compiler. Taking that as the
    # signature would collapse every distinct internal error into one group
    # and hide where the failure actually was.
    m = _ICE_RE.search(text)
    if m:
        where = f"{_short_path(m.group(1))}:{m.group(2)}"
        msg = m.group(3)

        # TINT_UNIMPLEMENTED is a *declaration* that a path was never
        # written — `default: TINT_IR_UNIMPLEMENTED(mod) << builtin.value()`
        # and its kin. Reaching one means the input used a feature tint
        # does not support yet, which is the documented behaviour, not a
        # defect. Filing them would fill the bug list with one entry per
        # unsupported builtin. TINT_ASSERT and TINT_UNREACHABLE are the
        # opposite: they assert something the compiler believes cannot
        # happen, so reaching one is always a real internal error.
        if msg.startswith("TINT_UNIMPLEMENTED"):
            return {"kind": "unimplemented", "signature": None,
                    "is_bug": False}

        kind_word = "UNREACHABLE" if msg.startswith("TINT_UNREACHABLE") else "ICE"
        return hit("ice", f"{kind_word}: {where}")

    # A genuine memory error carries no ICE line above it, so ordering the
    # ICE first cannot swallow one.
    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {_normalize(m.group(1))}")

    m = _SIGNAL_RE.search(text)
    if m:
        return hit("signal", m.group(1))

    # An ordinary diagnostic: the expected outcome of compiling a fused
    # (usually ill-formed) program, and not a finding.
    if _DIAGNOSTIC_RE.search(text):
        return {"kind": "diagnostic", "signature": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# A WGSL entry point is what a backend actually compiles; a shader with no
# entry point validates but exercises none of the code writers. tint marks
# them with a stage attribute.
_ENTRY_RE = re.compile(r'@(vertex|fragment|compute)\b')

# `enable`/`requires` directives switch on optional language features
# (f16, subgroups, ...). They must stay at the top of the file, before any
# declaration — a fused program that buries one produces a spurious error.
_ENABLE_RE = re.compile(r'^\s*(?:enable|requires)\s+([A-Za-z0-9_,\s]+);', re.M)

# Pipeline-overridable constants. A shader using them needs values supplied
# on the command line (--overrides) or it fails at the code-writer stage,
# which is a property of the invocation, not a bug.
_OVERRIDE_RE = re.compile(r'^\s*override\b', re.M)

# Constructs that reach the parts most likely to be wrong: pointers and
# references (WGSL's memory model), atomics, and workgroup storage.
_MEMORY_RE = re.compile(
    r'\b(?:ptr\s*<|atomic\s*<|workgroup\b|storage\b|var\s*<)')
_TEXTURE_RE = re.compile(r'\btexture_(?:2d|3d|cube|storage|depth|multisampled)')


def analyze_seed(content, filename=""):
    """Facts about a tint seed that the driver needs.

    Returns a dict:
        entry_stages   the shader stages present (vertex/fragment/compute);
                       a backend only compiles an entry point, so a shader
                       with none exercises only the front end
        has_entry      whether any entry point exists
        enables        optional-feature directives it declares
        uses_overrides needs pipeline-override values on the command line
        memory_score   pointers, atomics, workgroup/storage vars
        texture_score  texture types, a dense source of backend-specific code
    """
    text = content or ""
    stages = sorted(set(_ENTRY_RE.findall(text)))
    enables = []
    for m in _ENABLE_RE.finditer(text):
        enables.extend(e.strip() for e in m.group(1).split(",") if e.strip())
    return {
        "entry_stages": stages,
        "has_entry": bool(stages),
        "enables": list(dict.fromkeys(enables)),
        "uses_overrides": bool(_OVERRIDE_RE.search(text)),
        "memory_score": len(_MEMORY_RE.findall(text)),
        "texture_score": len(_TEXTURE_RE.findall(text)),
    }
