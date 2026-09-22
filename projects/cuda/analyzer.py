"""
projects/cuda/analyzer.py — decide what a clang-CUDA or nvcc run means.

Two compilers, two output dialects:

  clang   LLVM's crash handler ("PLEASE submit a bug report", "Stack
          dump:" with frames), `Assertion ... failed`, `UNREACHABLE
          executed`, `LLVM ERROR:`, `fatal error: error in backend`, or a
          bare signal. Diagnostics are `file:line:col: error: ...`.
  nvcc    a driver over cudafe++ (EDG frontend), cicc (LLVM-based device
          compiler) and ptxas. Crashes surface as
          `nvcc error   : 'cicc' died due to signal 11`,
          `nvcc error   : 'ptxas' died ...`, `Internal Compiler Error`
          (EDG), `catastrophic error:` (EDG), `ptxas fatal   :`, or an
          assertion from cicc's LLVM. Diagnostics are `file(line): error:`.

What is *not* a finding: resource exhaustion under the address-space cap
(LLVM reports it as `LLVM ERROR: out of memory`, nvcc as `cicc` dying
with signal 9 or "Cannot allocate memory"), nvcc refusing a flag/arch
combination ("unsupported gpu architecture", "not supported"), and
clang's "unknown CUDA version" warning.
"""

import re

_VOLATILE = [
    (re.compile(r"/tmp/\S+"), "<tmp>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d{4,}\b"), "N"),
]


def _normalize(text):
    for rx, rep in _VOLATILE:
        text = rx.sub(rep, text)
    return text.strip()


_SOURCE_ECHO_RE = re.compile(r"^\s*\d+\s*\|.*$", re.M)

_RESOURCE_RE = re.compile(
    r"out of memory|std::bad_alloc|Cannot allocate memory|Out of memory|"
    r"^Killed$|hard rss limit|died due to signal 9\b|"
    r"failed to allocate|stack overflow|Stack overflow|"
    r"virtual memory exhausted|resource temporarily unavailable",
    re.I | re.M)

_UNSUPPORTED_RE = re.compile(
    r"unsupported gpu architecture|is not supported|not supported for|"
    r"Unsupported CUDA version|cannot find libdevice|"
    r"No such file or directory|unrecognized command-line option|"
    r"unknown argument|unsupported option|Unknown option",
    re.I)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_ASSERT_RE = re.compile(r"([\w./\-]+):(\d+):\s.*?Assertion `(.+?)' failed", re.S)
_UNREACHABLE_RE = re.compile(r"UNREACHABLE executed(?: at ([\w./\-]+):(\d+))?")
_LLVM_ERROR_RE = re.compile(r"LLVM ERROR:\s*([^\n]+)")
_BACKEND_RE = re.compile(r"fatal error: error in backend:\s*([^\n]+)")
_NVCC_DIED_RE = re.compile(r"nvcc error\s*:\s*'(\w+)' died (?:due to|with) signal (\d+)")
_NVCC_ICE_RE = re.compile(r"(Internal Compiler Error|catastrophic error:|ptxas fatal\s*:|"
                          r"nvcc error\s*:)\s*([^\n]*)")
_SIGNAL_RE = re.compile(r"^(Segmentation fault|Aborted|Illegal instruction|"
                        r"Bus error|Floating point exception|Trace/breakpoint trap)",
                        re.M)
_CLANG_ERROR_RE = re.compile(r"^[^\n]*?:\d+:\d+: (?:fatal )?error: ", re.M)
_NVCC_ERROR_RE = re.compile(r"^[^\n]*?\(\d+\): (?:catastrophic )?error: |"
                            r"^[^\n]*?: error: |\d+ errors? detected in the compilation",
                            re.M)
_STACK_MSG_RE = re.compile(r"^\d+\.\t(?:\S+:\d+:\d+:\s*)?(.+)$", re.M)
_STACK_FRAME_RE = re.compile(r"^\s*\d+\s+\S+\s+0x[0-9a-f]+\s+([A-Za-z_][\w:<>,~ &*]*?)\s*\(", re.M)
_NOISE_FRAME_RE = re.compile(r"^(?:llvm::sys::PrintStackTrace|llvm::sys::RunSignalHandlers|"
                             r"llvm::sys::CleanupOnSignal|.*SignalHandler.*|abort|raise|"
                             r"gsignal|pthread_kill|__assert_fail|__cxa_throw)$")


def _short(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def _stack_fingerprint(text):
    """Crash site from LLVM's pretty stack trace: the diagnostic line after
    'Program arguments' (per-file prefix stripped) plus the first real
    frames, skipping the signal-handler noise."""
    m = re.search(r"Stack dump:\n((?:.*\n?){1,80})", text)
    if not m:
        return None
    body = m.group(1)
    msgs = [x for x in _STACK_MSG_RE.findall(body) if not x.startswith("Program arguments")]
    frames = [f for f in _STACK_FRAME_RE.findall(body) if not _NOISE_FRAME_RE.match(f)]
    parts = []
    if msgs:
        parts.append(_normalize(msgs[-1])[:70])
    if frames:
        parts.append(" < ".join(f[:40] for f in frames[:3]))
    return " | ".join(parts) or None


def is_resource_exhaustion(output):
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output, tool="clang", return_code=None):
    """{'kind', 'signature', 'is_bug', 'is_valid'} for one compiler run.
    kind: sanitizer | assert | unreachable | llvm_error | backend |
          nvcc_crash | signal | crash | timeout | resource | rejected |
          unsupported | ok."""
    out = _SOURCE_ECHO_RE.sub("", output or "")

    def hit(kind, signature):
        sig = signature
        if tool == "nvcc":
            sig = f"{sig} [nvcc]"
        return {"kind": kind, "signature": sig, "is_bug": True, "is_valid": True}

    if return_code == 124 or "Timeout" in out[:200] and return_code in (124, -9):
        return {"kind": "timeout", "signature": None, "is_bug": False, "is_valid": False}
    if is_resource_exhaustion(out):
        return {"kind": "resource", "signature": None, "is_bug": False, "is_valid": False}

    m = _ASAN_RE.search(out)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))[:80]}")
    m = _ASSERT_RE.search(out)
    if m:
        return hit("assert", f"Assertion {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _UNREACHABLE_RE.search(out)
    if m:
        where = f"{_short(m.group(1))}:{m.group(2)}" if m.group(1) else "?"
        return hit("unreachable", f"UNREACHABLE {where}")
    m = _BACKEND_RE.search(out)
    if m:
        return hit("backend", f"backend: {_normalize(m.group(1))[:80]}")
    m = _LLVM_ERROR_RE.search(out)
    if m:
        return hit("llvm_error", f"LLVM ERROR: {_normalize(m.group(1))[:80]}")
    m = _NVCC_DIED_RE.search(out)
    if m:
        return hit("nvcc_crash", f"{m.group(1)} died signal {m.group(2)}")
    if tool == "nvcc":
        m = _NVCC_ICE_RE.search(out)
        if m and not _UNSUPPORTED_RE.search(m.group(2)):
            kind_text = m.group(1).strip().rstrip(":").strip()
            # EDG's "catastrophic error" is mostly a user error it cannot
            # recover from (missing include, too many errors) and ptxas's
            # "fatal" is mostly an input problem (unresolved extern
            # without -rdc, unsupported .version); only an internal
            # failure in either is a finding.
            if kind_text.startswith(("catastrophic", "ptxas fatal")) and \
                    not re.search(r"internal|assert|signal|died|crash|unexpected", m.group(2), re.I):
                return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
            if kind_text.startswith("nvcc error"):
                # nvcc's own driver errors ("nvcc error : 'x' returned
                # non-zero status") wrap a child's diagnostic; a genuine
                # rejection sits above it as an EDG/cicc error line.
                if _NVCC_ERROR_RE.search(out) or _CLANG_ERROR_RE.search(out):
                    return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
            return hit("nvcc_crash", f"{kind_text}: {_normalize(m.group(2))[:70]}")
    fp = _stack_fingerprint(out)
    if fp or "PLEASE submit a bug report" in out:
        return hit("crash", f"Stack dump: {fp or 'no frames'}")
    m = _SIGNAL_RE.search(out)
    if m:
        return hit("signal", f"{m.group(1)} ({tool})")
    if return_code is not None and return_code < 0:
        return hit("signal", f"signal {-return_code} ({tool})")
    if return_code is not None and return_code >= 128 and return_code != 124:
        return hit("signal", f"signal {return_code - 128} ({tool})")

    if _UNSUPPORTED_RE.search(out) and return_code not in (0, None):
        return {"kind": "unsupported", "signature": None, "is_bug": False, "is_valid": False}
    if _CLANG_ERROR_RE.search(out) or _NVCC_ERROR_RE.search(out) or \
            (return_code not in (0, None)):
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    return {"kind": "ok", "signature": None, "is_bug": False, "is_valid": True}


def crash_signature(output, tool="clang", return_code=None):
    return classify(output, tool, return_code)["signature"]
