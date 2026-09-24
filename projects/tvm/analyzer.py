"""
projects/tvm/analyzer.py — decide what one runner.py execution means.

runner.py prints a marker for the outcomes it can see from Python:
FFL_REJECTED (the seed is not a valid program: TVMScript diagnostics,
verification errors, unsupported constructs, Python errors inside the
seed), FFL_INTERNAL_ERROR (a `Check failed` / InternalError — TVM's own
invariant, ICHECK, tripped), FFL_MISMATCH (opt-level/target outputs
disagree), FFL_OK. What it cannot see kills the process first: a
segfault in libtvm or LLVM, an LLVM assertion (the linked LLVM is the
assertion-enabled build), `LLVM ERROR`, an abort.

Resource exhaustion under the address-space cap and timeouts are not
findings. A Python traceback without a marker is a runner error (an
uncaught exception in the harness itself) and is reported as such so it
is not mistaken for a compiler bug.
"""

import re

_VOLATILE = [
    (re.compile(r"function \S+ already exists"), "function <name> already exists"),
    (re.compile(r"In PrimFunc \S+ variables \([^)]*\)"), "In PrimFunc <f> variables (<vars>)"),
    (re.compile(r"Buffer \S+ is used before"), "Buffer <name> is used before"),
    (re.compile(r"of buffer \S+ \("), "of buffer <name> ("),
    (re.compile(r"/tmp/\S+"), "<tmp>"),
    (re.compile(r"\.fused/\S+"), "<tmp>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d{4,}\b"), "N"),
]


def _normalize(text):
    for rx, rep in _VOLATILE:
        text = rx.sub(rep, text)
    return text.strip()


_RESOURCE_RE = re.compile(r"out of memory|std::bad_alloc|MemoryError|Cannot allocate memory|^Killed$|"
                          r"failed to allocate|stack overflow|Stack overflow|RecursionError", re.I | re.M)
_CHECK_RE = re.compile(r"(?:InternalError|Check failed)[^\n]*?(?:\(([\w./\-]+):(\d+)\))?[^\n]*?Check failed:\s*([^\n]{0,80})")
_CHECK_LOC_RE = re.compile(r"([\w./\-]+\.(?:cc|h|cpp)):(\d+)[^\n]*Check failed:\s*([^\n]{0,80})")
_LLVM_ASSERT_RE = re.compile(r"([\w./\-]+):(\d+):\s.*?Assertion `(.+?)' failed", re.S)
_UNREACHABLE_RE = re.compile(r"UNREACHABLE executed(?: at ([\w./\-]+):(\d+))?")
_LLVM_ERROR_RE = re.compile(r"LLVM ERROR:\s*([^\n]+)")
_SAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_SIGNAL_RE = re.compile(r"^(Segmentation fault|Aborted|Illegal instruction|Bus error|Floating point exception)", re.M)
_MISMATCH_RE = re.compile(r"FFL_MISMATCH ([^\n]+)")
#: `TVM_FFI_THROW(InternalError) << "msg"` and `ICHECK_EQ` render without
#: the "Check failed:" prefix the pattern above needs; the Python side
#: shows them as `tvm.error.InternalError: <msg>` (last line of the
#: traceback). Without this the signature came out as bare "ICHECK".
_TVM_ERROR_RE = re.compile(r"^tvm\.error\.(\w+): ([^\n]{1,160})", re.M)
_STACK_TOP_RE = re.compile(r"^\s*\d+:\s+(?:0x[0-9a-f]+\s+)?(tvm::[\w:<>]+)", re.M)


def _short(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def is_resource_exhaustion(output):
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output, tool="build", return_code=None):
    out = output or ""

    def hit(kind, sig):
        return {"kind": kind, "signature": f"{sig} [{tool}]" if tool not in ("build", None) else sig,
                "is_bug": True, "is_valid": True}

    if return_code == 124:
        return {"kind": "timeout", "signature": None, "is_bug": False, "is_valid": False}
    if is_resource_exhaustion(out):
        return {"kind": "resource", "signature": None, "is_bug": False, "is_valid": False}
    m = _SAN_RE.search(out)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))[:80]}")
    m = _LLVM_ASSERT_RE.search(out)
    if m:
        return hit("assert", f"Assertion {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _UNREACHABLE_RE.search(out)
    if m:
        return hit("unreachable", f"UNREACHABLE {_short(m.group(1)) + ':' + m.group(2) if m.group(1) else '?'}")
    m = _LLVM_ERROR_RE.search(out)
    if m:
        return hit("llvm_error", f"LLVM ERROR: {_normalize(m.group(1))[:80]}")
    if "FFL_REJECTED" in out:
        # the runner's verdict wins: a rejection message may quote an
        # ICHECK text (a parser diagnostic, a pass precondition)
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    if "FFL_INTERNAL_ERROR" in out or "Check failed" in out:
        m = _CHECK_LOC_RE.search(out)
        where = f"{_short(m.group(1))}:{m.group(2)} " if m else ""
        msg = m.group(3) if m else (re.search(r"Check failed:\s*([^\n]{0,80})", out).group(1)
                                    if re.search(r"Check failed:", out) else "")
        if not msg:
            # a TVM_FFI_THROW / ICHECK_EQ message: no "Check failed:" text,
            # only the Python-side `tvm.error.InternalError: <msg>` line
            e = _TVM_ERROR_RE.search(out)
            if e:
                msg = e.group(2)
                if not where:
                    # the deepest TVM source frame named in the traceback is
                    # where it was thrown
                    frames = re.findall(r'File "[^"]*?/tvm/src/([\w/.\-]+)", line (\d+)', out)
                    if frames:
                        where = f"{_short(frames[-1][0])}:{frames[-1][1]} "
        return hit("check", f"ICHECK {where}{_normalize(msg)[:70]}".strip())
    m = _MISMATCH_RE.search(out)
    if m:
        text = re.sub(r"at \([^)]*\)", "at (i)", m.group(1))
        return hit("mismatch", f"Mismatch: {_normalize(text)[:90]}")
    m = _SIGNAL_RE.search(out)
    if m or (return_code is not None and (return_code < 0 or return_code >= 128)):
        sig = m.group(1) if m else f"signal {return_code - 128 if return_code >= 128 else -return_code}"
        top = _STACK_TOP_RE.search(out)
        return hit("signal", f"{sig}{' in ' + top.group(1)[:60] if top else ''}")
    if "FFL_REJECTED" in out:
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    if "FFL_OK" in out:
        return {"kind": "ok", "signature": None, "is_bug": False, "is_valid": True}
    if "Traceback (most recent call last)" in out:
        # the harness itself failed: report, but as its own class
        last = [l for l in out.splitlines() if l.strip()][-1] if out.strip() else ""
        return hit("runner_error", f"runner: {_normalize(last)[:80]}")
    if return_code not in (0, None):
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    return {"kind": "ok", "signature": None, "is_bug": False, "is_valid": True}


def crash_signature(output, tool="build", return_code=None):
    return classify(output, tool, return_code)["signature"]
