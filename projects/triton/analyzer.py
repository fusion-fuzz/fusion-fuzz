"""
projects/triton/analyzer.py — the Triton-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/triton/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what triton-opt printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a .mlir file that
    determine how it must be run, above all its pass pipeline.

triton-opt is a *compiler* tool, not an executor: the expected outcome of a
malformed input is a clean diagnostic (`<loc>: error: ...`), not a crash. A
fused MLIR module is usually ill-typed — two modules' tensor layouts rarely
agree — so ordinary diagnostics are the common case and not findings. The
oracle's job is to separate the assertion failures and crashes, which are
Triton's or MLIR's fault, from that flood.

What a triton-opt failure looks like
------------------------------------
The shapes below are taken from the LLVM/MLIR sources the build links
against, not from memory. Triton's own code raises failures three ways
(counted in lib/ at the pinned revision): 658 `assert`, 50
`llvm::report_fatal_error`, 35 `llvm_unreachable`.

    triton-opt: /path/Foo.cpp:123: void f(): Assertion `expr' failed.
        A C assert. Live only because the prebuilt LLVM this links against
        is configured with LLVM_ENABLE_ASSERTIONS=ON
        (scripts/build-llvm-project.sh in the Triton tree) — without it
        most of what this fuzzer looks for is silent.

    LLVM ERROR: <message>
        `llvm::report_fatal_error`. A condition the code decided was
        unrecoverable.

    UNREACHABLE executed at /path/Foo.cpp:123!
        `llvm_unreachable` — a state the author believed impossible.

    PLEASE submit a bug report to https://github.com/llvm/llvm-project ...
    Stack dump:
        LLVM's crash handler, printed for a signal. The stack dump is what
        makes two crashes distinguishable.

    <file>:12:34: error: 'ttg.convert_layout' op ...
        An ordinary MLIR diagnostic. Not a finding — this is what a
        malformed fused module is supposed to produce.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that mean the machine (or our own cap) ran out of something.
# Checked first: LLVM routes allocation failure through the same
# `LLVM ERROR:` channel as a genuine fatal error, so matching that channel
# without subtracting these would file every OOM as a compiler bug.
_RESOURCE_RE = re.compile(
    r"LLVM ERROR: out of memory"
    r"|Allocation failed"
    r"|std::bad_alloc"
    r"|hard rss limit exhausted"
    r"|out of memory"
    r"|^Killed$"
    r"|Stack overflow|stack overflow",
    re.IGNORECASE | re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_UBSAN_RE = re.compile(r"runtime error:\s*([^\n]+)")

# glibc's assert: "<prog>: <file>:<line>: <func>: Assertion `<expr>' failed."
_ASSERT_RE = re.compile(
    r"^[^\n]*?([\w./+-]+\.(?:cpp|cc|h|hpp|inc)):(\d+):[^\n]*?"
    r"Assertion\s+`([^']*)'\s+failed", re.M)

# llvm_unreachable
_UNREACHABLE_RE = re.compile(
    r"UNREACHABLE executed(?:\s+at\s+([\w./+-]+):(\d+))?", re.M)

# llvm::report_fatal_error
_LLVM_ERROR_RE = re.compile(r"^LLVM ERROR:\s*([^\n]+)", re.M)

# `LLVM ERROR:` is also how Triton reports a *precondition on the input*
# that the module does not meet — most often a missing module attribute.
# Fusing two modules drops the `module attributes {...}` wrapper of at
# least one of them, so the result legitimately lacks `ttg.num-warps` and
# the pass legitimately refuses to run. That is the tool working, not a
# defect: the same message appears if a user hand-writes a module without
# the attribute.
#
# These are recognised by shape rather than listed exhaustively — they all
# say what the module *should* contain.
_INPUT_PRECONDITION_RE = re.compile(
    r"LLVM ERROR:[^\n]*?(?:"
    r"should contain a [\w.-]+ attribute"
    r"|failed to lookup the number of warps"
    r"|module should contain"
    r"|requires .* attribute"
    r"|expected .* attribute on the module"
    r")", re.I)

# LLVM's crash handler. The first frame that is not the handler itself is
# the useful grouping key.
_CRASH_BANNER_RE = re.compile(r"PLEASE submit a bug report")
_FRAME_RE = re.compile(r"^\s*#?\d+\s+[\dxa-fA-F]*\s+(\S.*?)(?:\s*\+\s*\d+)?$", re.M)
_REPORTING_FRAMES = (
    "llvm::sys::PrintStackTrace", "llvm::sys::RunSignalHandlers",
    "llvm::sys::CleanupOnSignal", "SignalHandler", "__restore_rt",
    "abort", "raise", "llvm::report_fatal_error",
)

_SIGNAL_RE = re.compile(
    r"^(Segmentation fault|Bus error|Aborted|Illegal instruction|"
    r"Floating point exception)(?:\s*\(core dumped\))?\s*$", re.M)

# An ordinary MLIR diagnostic: `<file>:<line>:<col>: error: <msg>`. This is
# the expected outcome of running a fused (usually ill-typed) module.
_DIAGNOSTIC_RE = re.compile(r"^[^\n]*?:\d+:\d+:\s*error:\s", re.M)

# Detail that varies run to run and would defeat deduplication.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"^.*?/(?:lib|include)/"), ""),
    (re.compile(r"\b\d{6,}\b"), "N"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return " ".join(text.split())[:160]


def _short_path(path):
    """Keep the part of a source path that identifies the file.

    The build embeds absolute paths from whatever directory it ran in;
    only the tail below `lib/`, `include/` or the file name is stable
    across machines.
    """
    path = path.replace("\\", "/")
    for marker in ("/lib/", "/include/", "/triton/"):
        idx = path.find(marker)
        if idx >= 0:
            return path[idx + 1:]
    return path.rsplit("/", 1)[-1]


def _failing_frame(output):
    """The first stack frame that is not part of the crash reporting."""
    for frame in _FRAME_RE.findall(output or ""):
        frame = frame.strip()
        if not frame or frame.startswith(("0x", "/")):
            continue
        if any(frame.startswith(r) for r in _REPORTING_FRAMES):
            continue
        return _normalize(frame)
    return None


def is_resource_exhaustion(output):
    """True when triton-opt died for lack of memory or stack.

    A fused MLIR module can nest regions deeply enough to overflow the
    verifier's recursion; that is a property of the input, not a bug.
    """
    return bool(_RESOURCE_RE.search(output or ""))


def classify(output):
    """Categorise triton-opt's combined stdout+stderr.

    Returns a dict with:
        kind       "sanitizer", "ubsan", "assert", "unreachable",
                   "llvm_error", "crash", "signal", "resource",
                   "diagnostic", "clean"
        signature  stable grouping key, or None when not a finding
        is_bug     whether this should be saved

    Order is load-bearing: resource exhaustion is subtracted first,
    because LLVM reports allocation failure through the same
    `LLVM ERROR:` channel as a real fatal error; and every internal
    failure is recognised before the ordinary-diagnostic check, since a
    crash report also contains lines the diagnostic pattern would match.
    """
    text = output or ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature, "is_bug": True}

    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None, "is_bug": False}

    m = _ASAN_RE.search(text)
    if m:
        return hit("sanitizer", f"{m.group(1)}: {_normalize(m.group(2))}")
    m = _UBSAN_RE.search(text)
    if m:
        return hit("ubsan", f"UBSAN: {_normalize(m.group(1))}")

    # The assertion expression is the most specific thing available, and
    # file:line groups better than the message, which often embeds a value.
    m = _ASSERT_RE.search(text)
    if m:
        where = f"{_short_path(m.group(1))}:{m.group(2)}"
        return hit("assert", f"ASSERT: {where} ({_normalize(m.group(3))[:70]})")

    m = _UNREACHABLE_RE.search(text)
    if m:
        where = f"{_short_path(m.group(1))}:{m.group(2)}" if m.group(1) else "?"
        return hit("unreachable", f"UNREACHABLE: {where}")

    m = _LLVM_ERROR_RE.search(text)
    if m:
        # A precondition the fused module does not meet is not a defect;
        # see _INPUT_PRECONDITION_RE.
        if _INPUT_PRECONDITION_RE.search(text):
            return {"kind": "diagnostic", "signature": None, "is_bug": False}
        return hit("llvm_error", f"LLVM ERROR: {_normalize(m.group(1))}")

    if _CRASH_BANNER_RE.search(text):
        frame = _failing_frame(text)
        return hit("crash", f"CRASH: {frame or 'unknown frame'}")

    m = _SIGNAL_RE.search(text)
    if m:
        frame = _failing_frame(text)
        return hit("signal", f"{m.group(1)}{f' in {frame}' if frame else ''}")

    # An ordinary diagnostic: the expected outcome of running a fused
    # (usually ill-typed) module, and not a finding.
    if _DIAGNOSTIC_RE.search(text):
        return {"kind": "diagnostic", "signature": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# Every Triton test declares how it must be run:
#   // RUN: triton-opt %s -split-input-file -tritongpu-coalesce | FileCheck %s
#
# The pass pipeline is not decoration. `-tritongpu-pipeline=num-stages=3`
# *is* what the test exercises; running that IR through some other pipeline
# exercises nothing in particular. So the pipeline travels with the seed.
_RUN_RE = re.compile(r"^//\s*RUN:\s*(.+)$", re.M)

# Options that belong to the lit harness rather than to triton-opt, and
# everything after the first pipe (FileCheck's own arguments).
_HARNESS_OPTS = {
    "-split-input-file", "--split-input-file",   # driver splits sections itself
    "-verify-diagnostics", "--verify-diagnostics",
    "--check-prefix", "--check-prefixes", "--dump-input-context",
    "--implicit-check-not", "--allow-empty",
}

# A pass name, or a pass with its options: `-tritongpu-pipeline=num-stages=3`.
_PASS_RE = re.compile(r'^--?[a-zA-Z][\w-]*(?:=[^\s]+)?$')

# Layout aliases: `#blocked = #ttg.blocked<{...}>`. 247 of Triton's 297
# tests define at least one, and the IR body references them by name — so
# fusing two modules collides their aliases unless they are renamed.
_ALIAS_RE = re.compile(r'^#([A-Za-z_][\w]*)\s*=\s*#', re.M)

# `module attributes {...}` carries the target description
# (`ttg.target = "cuda:80"`, warp size, ...). Two fused modules disagree
# on it, and the passes read it.
_MODULE_ATTR_RE = re.compile(r'^module\s+attributes\s*\{([^}]*)\}', re.M)

_FUNC_RE = re.compile(r'^\s*(?:tt\.func|func\.func)\s+(?:public\s+|private\s+)?@([\w$.]+)', re.M)


def _parse_run_line(line):
    """Split a RUN line into (triton-opt flags, needs_split).

    Everything after the first `|` is FileCheck's, not ours. `%s` is lit's
    placeholder for the input file and is dropped — the driver supplies a
    real path.
    """
    cmd = line.split("|", 1)[0]
    toks = cmd.split()
    flags, needs_split = [], False
    seen_tool = False
    for tok in toks:
        if not seen_tool:
            if "triton-opt" in tok:
                seen_tool = True
            continue
        if tok in ("%s", "%t"):
            continue
        if tok.startswith("%") or tok.startswith("/") or tok.endswith(".mlir"):
            continue
        if tok in ("-split-input-file", "--split-input-file"):
            needs_split = True
            continue
        if tok in _HARNESS_OPTS:
            continue
        if _PASS_RE.match(tok):
            flags.append(tok)
    return flags, needs_split


# Generic-form ops — `"dialect.op"(...)`. A dialect prefix outside the set
# triton-opt registers is an *unregistered* op, which only parses under
# `-allow-unregistered-dialect`. MLIR's own analyses bail on these (a
# symbol-use query over a region holding one returns nullopt), so passes
# that assume the query succeeded abort rather than diagnose.
_GENERIC_OP_RE = re.compile(r'"([a-zA-Z_][\w]*)\.[\w.]+"\s*\(')

_REGISTERED_DIALECTS = frozenset((
    "tt", "ttg", "ttng", "tti", "nvws", "gluon", "proton", "proton_gpu",
    "amdgpu", "amdg", "nvg", "arith", "scf", "cf", "math", "llvm", "nvvm",
    "rocdl", "builtin", "ub", "memref", "vector", "gpu", "index",
))


def has_unregistered_ops(content):
    """True if the module uses an op from a dialect triton-opt does not
    register. Such a module is only parseable with
    `-allow-unregistered-dialect`."""
    return any(d not in _REGISTERED_DIALECTS
               for d in _GENERIC_OP_RE.findall(content or ""))


def analyze_seed(content, filename=""):
    """Facts about a Triton seed that the driver and the strategies need.

    Returns a dict:
        passes          the pass pipeline from the seed's RUN line. Without
                        it the seed is being run through the wrong compiler.
        needs_split     the RUN line used -split-input-file, so the file is
                        several independent modules joined by `// -----`
        aliases         layout alias names the IR defines and references
        module_attrs    the `module attributes {...}` payload, which
                        carries the target the passes read
        func_names      tt.func / func.func symbol names
        has_run_line    whether a pipeline was found at all
        unregistered    the module uses an unregistered dialect, so
                        passes that rely on symbol-use analysis are
                        not applicable to it
    """
    text = content or ""
    passes, needs_split = [], False
    m = _RUN_RE.search(text)
    if m:
        passes, needs_split = _parse_run_line(m.group(1))
    attrs = _MODULE_ATTR_RE.search(text)
    return {
        "passes": passes,
        "needs_split": needs_split,
        "aliases": _ALIAS_RE.findall(text),
        "module_attrs": attrs.group(1).strip() if attrs else None,
        "func_names": _FUNC_RE.findall(text),
        "has_run_line": bool(m),
        "unregistered": has_unregistered_ops(text),
    }
