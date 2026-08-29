"""
projects/gcc/analyzer.py — the GCC-specific analysis half of the adapter.

Two kinds of analysis live here, both consumed by projects/gcc/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what GCC printed is a bug, and if so what to call it.
  * Seed analysis (analyze_seed) — read the facts out of a seed's source
    that determine how it must be compiled.

They are separated from driver.py because they are pure functions of text:
they need no compiler, no temp directory and no subprocess, so they can be
tested directly and reasoned about without running anything.

Why the output half is not just a pattern list
----------------------------------------------
config.yaml's crash_patterns can only say "this string appeared". That is
enough to notice something went wrong, and not enough to be useful, because
GCC's failure output overloads the same strings across three very different
situations:

    internal compiler error: in tsubst_expr, at cp/pt.cc:21203
        A real ICE. An internal invariant broke. This is the finding.

    internal compiler error: Killed (program cc1plus)
        The OOM killer took the compiler. This is a fact about the machine.

    error: expected ';' before '}' token
        The fused program is ill-formed. This is the expected case — most
        fused programs are — and not interesting at all.

classify() draws those lines, and crash_signature() then produces a stable
grouping key so that a thousand hits on one broken invariant deduplicate to
one entry in outputs/ rather than a thousand.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that match crash_patterns but are about the machine, not about GCC.
# Reporting these buries the real findings under noise that no GCC maintainer
# can act on.
_RESOURCE_RE = re.compile(
    r"internal compiler error:\s*(?:Killed|Terminated)"      # OOM killer / SIGTERM
    r"|out of memory allocating"                             # our own ulimit -v
    r"|cc1(?:plus)?: out of memory"
    r"|virtual memory exhausted"
    r"|memory exhausted"
    r"|hard rss limit exhausted"                              # ASan's own cap
    r"|No space left on device"
    r"|Segmentation fault \(program (?:collect2|ld|as)\)",   # binutils, not GCC
    re.IGNORECASE)

# "internal compiler error: in tsubst_expr, at cp/pt.cc:21203"
_ICE_AT_RE = re.compile(
    r"internal compiler error:\s*in\s+([\w:~<>()]+),\s*at\s+([\w./+-]+:\d+)")
# "internal compiler error: Segmentation fault"
_ICE_PLAIN_RE = re.compile(r"internal compiler error:\s*([^\n]+)")
# "during GIMPLE pass: vrp" / "during RTL pass: expand" — GCC names the pass
# that failed, which is the most useful grouping key it gives us: the same
# invariant broken from two different passes is two different bugs.
_PASS_RE = re.compile(r"during (\w+) pass:\s*(\S+)")
# --enable-checking=yes,extra,rtl failures.
_VERIFY_RE = re.compile(r"(verify_\w+ failed)|((?:tree|GIMPLE|RTL) check:[^\n]*)")

_ASAN_WITH_FRAME_RE = re.compile(r"SUMMARY: AddressSanitizer:\s+(\S+).*?in (\S+)", re.S)
_ASAN_RE = re.compile(r"SUMMARY: AddressSanitizer:\s+([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s+([^\n]+)")


def is_resource_exhaustion(output):
    """True when the compiler died for lack of memory, disk or time.

    Not a bug: the same input on a bigger machine compiles fine, so there is
    nothing to report and nothing to reduce.
    """
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output):
    """Categorise GCC's combined stdout+stderr.

    Returns a dict with:
        kind       one of "asan", "ubsan", "ice", "checking", "signal",
                   "resource", "diagnostic", "clean"
        signature  stable grouping key, or None when kind is not a finding
        gcc_pass   the failing pass ("GIMPLE:vrp"), when GCC named one
        is_bug     whether this should be saved as a finding

    The order below is deliberate: a sanitizer report is more precise than
    the ICE line that follows it, and a resource failure must be recognised
    before the ICE text it also prints.
    """
    text = output or ""
    gcc_pass = None
    p = _PASS_RE.search(text)
    if p:
        gcc_pass = f"{p.group(1)}:{p.group(2)}"
    suffix = f" [{gcc_pass}]" if gcc_pass else ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "gcc_pass": gcc_pass,
                "is_bug": True}

    # A sanitizer report pins the exact bad access inside the compiler, which
    # is strictly more information than the ICE line GCC prints afterwards.
    m = _ASAN_WITH_FRAME_RE.search(text)
    if m:
        return hit("asan", f"ASAN: {m.group(1)} in {m.group(2)}")
    m = _ASAN_RE.search(text)
    if m:
        return hit("asan", f"ASAN: {m.group(1).strip()[:120]}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {m.group(1).strip()[:120]}")

    # Before any ICE handling: the machine ran out of something.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "gcc_pass": gcc_pass,
                "is_bug": False}

    # An ICE with a source location is self-grouping: same function and line
    # means the same internal invariant broke.
    m = _ICE_AT_RE.search(text)
    if m:
        return hit("ice", f"ICE: in {m.group(1)}, at {m.group(2)}{suffix}")

    # A GIMPLE/RTL verifier failure names the broken invariant directly.
    m = _VERIFY_RE.search(text)
    if m:
        detail = (m.group(1) or m.group(2)).strip()[:120]
        return hit("checking", f"ICE: {detail}{suffix}")

    m = _ICE_PLAIN_RE.search(text)
    if m:
        return hit("ice", f"ICE: {m.group(1).strip()[:120]}{suffix}")

    # A signal with no ICE line means GCC died before its own handler could
    # print anything — usually a stack overflow or a crash inside a plugin.
    # Worth keeping, but there is nothing to group on beyond the signal.
    if "Segmentation fault" in text:
        return hit("signal", f"Segmentation fault (no ICE line){suffix}")
    if "Aborted" in text:
        return hit("signal", f"Aborted (no ICE line){suffix}")

    if "error: " in text or "fatal error: " in text:
        return {"kind": "diagnostic", "signature": None, "gcc_pass": gcc_pass,
                "is_bug": False}
    return {"kind": "clean", "signature": None, "gcc_pass": gcc_pass,
            "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

CXX_EXTENSIONS = (".cc", ".cpp", ".cxx", ".C", ".ii", ".hpp")

# GCC's testsuite drives itself with DejaGnu directives in comments:
#     /* { dg-do compile } */
#     /* { dg-options "-O2 -fno-tree-vectorize" } */
# The options are part of what the test is *for* — a vectoriser test
# compiled without -ftree-vectorize exercises nothing — so they are worth
# honouring rather than compiling the body under unrelated flags.
_DG_OPTIONS_RE = re.compile(r'dg-(?:additional-)?options\s+"([^"]*)"')
_DG_DO_RE = re.compile(r'dg-do\s+(\w+)')
# Only flags that cannot make the *driver* fail rather than the compiler:
# no -m<arch> (needs matching libraries), no -I/-L (they pointed into the
# testsuite tree the seed no longer sits in).
_DG_SAFE_RE = re.compile(r'^-(?:O[0-3sgz]?|f[\w=-]+|std=[\w+]+|W[\w-]+|g\d?)$')

# C++ constructs that mean a .c-extensioned seed must still go to g++ —
# fusion mixes corpora, so a child can inherit C++ from one parent and a
# .c extension from the other.
_CXX_MARKER_RE = re.compile(
    r'(?m)^\s*(?:template\s*<|class\s+\w+|namespace\s+\w*\s*\{|using\s+namespace\b)'
    r'|^\s*#\s*include\s*<(?:iostream|vector|string|map|memory|algorithm)>'
    r'|\bstd::')


def analyze_seed(content, extension=".c"):
    """Facts about a seed that determine how it must be compiled.

    Returns a dict:
        is_cxx        compile with g++ rather than gcc
        dg_options    the safe subset of the seed's dg-options, if any
        dg_do         "compile" / "run" / "assemble" / ... if declared
        is_dejagnu    whether the seed carries DejaGnu directives at all
    """
    text = content or ""
    is_cxx = extension in CXX_EXTENSIONS or bool(_CXX_MARKER_RE.search(text))

    dg_options = []
    m = _DG_OPTIONS_RE.search(text)
    if m:
        dg_options = [o for o in m.group(1).split() if _DG_SAFE_RE.match(o)][:4]

    m = _DG_DO_RE.search(text)
    dg_do = m.group(1) if m else None

    return {"is_cxx": is_cxx, "dg_options": dg_options, "dg_do": dg_do,
            "is_dejagnu": "{ dg-" in text}
