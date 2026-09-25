"""
projects/mojo/analyzer.py — decide what a `mojo build` / `mojo run` run
means.

The compiler (the `mojo` driver embedding the parser, the KGEN/MLIR
pipeline and LLVM) reports an internal failure with its own banner —
"Please submit a bug report to https://github.com/modular/modular/issues
and include the crash backtrace" — followed by an LLVM-style stack dump;
an assertions-on build also prints `Assertion ... failed` / `UNREACHABLE
executed`, and the MLIR interpreter prints `INTERNAL ERROR:`. Ordinary
rejections are `file:line:col: error: ...` diagnostics.

`mojo run` also executes the program, whose own failures must not be
mistaken for compiler bugs: `Assert Error: ...` (the stdlib's debug_assert
under -D ASSERT=all) prints a stack dump too, `Unhandled exception` is a
raised error escaping main, and a plain non-zero exit is the test's own
verdict. A signal from the *program* (segfault after the build succeeded)
is kept as a separate, lower-confidence class: a fused program can misuse
an UnsafePointer, so it is a miscompile candidate, not a proven one.
"""

import re

_VOLATILE = [
    (re.compile(r"/tmp/\S+"), "<tmp>"),
    (re.compile(r"\.fused/\S+"), "<tmp>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b\d{4,}\b"), "N"),
]


def _normalize(text):
    for rx, rep in _VOLATILE:
        text = rx.sub(rep, text)
    return text.strip()


_RESOURCE_RE = re.compile(
    r"out of memory|std::bad_alloc|Cannot allocate memory|Out of memory|^Killed$|"
    r"failed to allocate|stack overflow|Stack overflow|hard rss limit|"
    r"virtual memory exhausted|resource temporarily unavailable", re.I | re.M)
_CRASH_BANNER_RE = re.compile(r"[Pp]lease submit a bug report", re.I)
#: glibc's allocator refusing to continue. These are memory-safety
#: findings of their own — a compiler that double-frees or writes past a
#: chunk header is not merely crashing — and they carry no stack frames,
#: so without this they were all filed as "compiler crash: no frames".
_GLIBC_RE = re.compile(
    r"^(double free or corruption[^\n]*|malloc\(\): [^\n]+|free\(\): [^\n]+|"
    r"realloc\(\): [^\n]+|munmap_chunk\(\): [^\n]+|corrupted (?:size vs\. prev_size|"
    r"double-linked list)[^\n]*|malloc_consolidate\(\)[^\n]*)", re.M)
# One line, and the file has to look like a compiler source file. With
# re.S and a dot-matches-newline `.*?` this used to start from a *source*
# position printed earlier in the diagnostic — `foo.mojo:600:5: note:` —
# and produce "Assertion 600:5 llvm::hasSingleElement(region)", a fresh
# class per input line. One defect became 12 bundles in a 55-minute batch.
_ASSERT_RE = re.compile(
    r"^[^\n]*?([\w./\-]+\.(?:cpp|cc|cxx|h|hpp|inc)):(\d+):[^\n]*?Assertion `(.+?)' failed",
    re.M)
_UNREACHABLE_RE = re.compile(r"UNREACHABLE executed(?: at ([\w./\-]+):(\d+))?")
_LLVM_ERROR_RE = re.compile(r"LLVM ERROR:\s*([^\n]+)")
_INTERNAL_RE = re.compile(r"(?:INTERNAL ERROR|internal error):\s*([^\n]+)")
_SAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_SIGNAL_RE = re.compile(r"^(Segmentation fault|Aborted|Illegal instruction|Bus error|"
                        r"Floating point exception|Trace/breakpoint trap)", re.M)
_RUNTIME_ASSERT_RE = re.compile(r"Assert Error:|Unhandled exception caught|^Error: |^ABORT: |\bABORT:", re.M)
_DIAG_ERROR_RE = re.compile(r"^[^\n]*?:\d+:\d+: error: |^[^\n]*mojo: error: ", re.M)
_STACK_MSG_RE = re.compile(r"^\d+\.\t(?:\S+:\d+:\d+:\s*)?(.+)$", re.M)
_STACK_FRAME_RE = re.compile(r"^\s*#?\d+\s+\S+\s+0x[0-9a-f]+\s+([A-Za-z_][\w:<>,~ &*()]*?)(?:\s*\+\s*\d+)?\s*$", re.M)
_NOISE_RE = re.compile(r"PrintStackTrace|RunSignalHandlers|SignalHandler|CleanupOnSignal|^abort$|^raise$|"
                       r"__assert_fail|__restore_rt|CrashReporting|crashpad", re.I)
_UNSUPPORTED_RE = re.compile(r"unsupported target|not supported on this|unknown target|"
                             r"cannot find libdevice|no such accelerator|unsupported CPU", re.I)


def _short(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def _stack_fingerprint(text):
    m = re.search(r"Stack dump[^\n]*\n((?:.*\n?){1,60})", text)
    if not m:
        return None
    body = m.group(1)
    msgs = [x for x in _STACK_MSG_RE.findall(body) if not x.startswith("Program arguments")]
    frames = [f for f in _STACK_FRAME_RE.findall(body) if not _NOISE_RE.search(f)]
    parts = []
    if msgs:
        parts.append(_normalize(msgs[-1])[:70])
    if frames:
        parts.append(" < ".join(f[:40] for f in frames[:3]))
    return " | ".join(parts) or None


def is_resource_exhaustion(output):
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output, tool="build", return_code=None, compiled_ok=None, program_may_trap=False):
    """{'kind', 'signature', 'is_bug', 'is_valid'}.

    tool: "build" (compile only), "run" (compile + execute), "parse"
    (kgen-translate). compiled_ok: for "run", whether the compile step
    succeeded (the driver knows; it decides whether a signal came from the
    compiler or the program)."""
    out = output or ""

    def hit(kind, sig):
        if tool == "parse":
            sig = f"{sig} [parse]"
        return {"kind": kind, "signature": sig, "is_bug": True, "is_valid": True}

    if return_code == 124:
        return {"kind": "timeout", "signature": None, "is_bug": False, "is_valid": False}
    if is_resource_exhaustion(out):
        return {"kind": "resource", "signature": None, "is_bug": False, "is_valid": False}

    m = _SAN_RE.search(out)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))[:80]}")
    m = _ASSERT_RE.search(out)
    if m:
        return hit("assert", f"Assertion {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _UNREACHABLE_RE.search(out)
    if m:
        where = f"{_short(m.group(1))}:{m.group(2)}" if m.group(1) else "?"
        return hit("unreachable", f"UNREACHABLE {where}")
    m = _LLVM_ERROR_RE.search(out)
    if m:
        return hit("llvm_error", f"LLVM ERROR: {_normalize(m.group(1))[:80]}")
    m = _INTERNAL_RE.search(out)
    if m:
        return hit("internal", f"internal error: {_normalize(m.group(1))[:80]}")
    m = _GLIBC_RE.search(out)
    if m:
        return hit("memory", f"glibc: {_normalize(m.group(1))[:70]}")
    if _CRASH_BANNER_RE.search(out):
        return hit("crash", f"compiler crash: {_stack_fingerprint(out) or 'no frames'}")

    # `abort()`, `os.abort`, `unreachable` and `debug_assert` all end in a
    # trap instruction (SIGILL) on x86: a program that spells one of those
    # is reporting its own verdict when it dies that way.
    runtime_own = bool(_RUNTIME_ASSERT_RE.search(out)) or program_may_trap
    m = _SIGNAL_RE.search(out)
    if m or (return_code is not None and (return_code < 0 or return_code >= 128)) :
        sig = m.group(1) if m else f"signal {return_code - 128 if return_code >= 128 else -return_code}"
        if tool == "run" and compiled_ok:
            if runtime_own:
                # debug_assert / raised error: the program's verdict
                return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
            return hit("runtime_signal", f"program {sig} [runtime]")
        return hit("signal", f"{sig} (compiler)")

    if _UNSUPPORTED_RE.search(out) and return_code not in (0, None):
        return {"kind": "unsupported", "signature": None, "is_bug": False, "is_valid": False}
    if _DIAG_ERROR_RE.search(out) or runtime_own or (return_code not in (0, None)):
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    return {"kind": "ok", "signature": None, "is_bug": False, "is_valid": True}


def crash_signature(output, tool="build", return_code=None):
    return classify(output, tool, return_code)["signature"]
