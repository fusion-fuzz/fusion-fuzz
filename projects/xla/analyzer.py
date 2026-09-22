"""
projects/xla/analyzer.py — the XLA-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/xla/driver.py (and the
seed half by projects/xla/parser.py):

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what run_hlo_module or hlo-opt printed is a bug, and
    what to call it.
  * Seed analysis (analyze_seed) — the facts about an HLO module that
    decide how it can be run and how it can be fused: the computations,
    the entry's parameters and their shapes, the opcodes it uses.

XLA is a compiler *and* an executor here, and the two tools give two
oracles:

  crash oracle       CHECK / DCHECK / assert failures, LLVM fatal errors,
                     signals, sanitizer reports. The build keeps DCHECK and
                     assert() live (setup.py), so a broken invariant in a
                     pass or in the LLVM backend surfaces as a
                     `Check failed:` line followed by the abort.

  status oracle      XLA's compiler reports most failures as an
                     absl::Status. `INVALID_ARGUMENT` (the verifier or the
                     parser rejecting the module) and `UNIMPLEMENTED` (an op
                     the CPU backend or the evaluator does not support) are
                     the expected outcomes of a fused, usually ill-typed
                     module. `INTERNAL` is the compiler saying *it* went
                     wrong — `TF_RET_CHECK` failures, "unexpected" states —
                     and is filed as a finding.

  differential       run_hlo_module runs the compiled CPU executable and
                     the HLO evaluator on the same random inputs and
                     compares. A `first mismatch at array index` with the
                     CPU result differing from the reference is a
                     miscompilation candidate. It is filed separately from
                     crashes and needs human triage (fast-math flags are
                     drawn by the driver, and a module that does
                     floating-point reductions can legitimately differ —
                     the driver disables fast-math when it asks for the
                     comparison, and integer/pred mismatches are the ones
                     worth chasing first).

What is *not* a finding: resource exhaustion (a fused module can declare
a huge shape), our own RSS cap, a timeout, and the deliberate
UNIMPLEMENTED/INVALID_ARGUMENT statuses above.
"""

import re

try:
    from core.hlo_text import parse_module, normalize_shape
except ImportError:  # pragma: no cover
    import importlib.util as _ilu, os as _os
    _spec = _ilu.spec_from_file_location(
        "ffl_hlo_text", _os.path.join(_os.path.dirname(_os.path.dirname(
            _os.path.dirname(_os.path.abspath(__file__)))), "core", "hlo_text.py"))
    _m = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_m)
    parse_module, normalize_shape = _m.parse_module, _m.normalize_shape

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

_RESOURCE_RE = re.compile(
    r"out of memory|std::bad_alloc|Allocation of \d+ bytes failed|"
    r"RESOURCE_EXHAUSTED|hard rss limit exhausted|^Killed$|"
    r"Cannot allocate memory|Out of memory|exceeds.*memory limit|"
    r"failed to allocate|stack overflow|Stack overflow|"
    # env.cc:93 CHECK when pthread_create fails: the address-space cap
    # (`ulimit -v`) or the pid limit, not the module
    r"creation via pthread_create",
    re.IGNORECASE | re.M)

_ASAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer):\s*([^\n]+)")
_ASAN_KIND_RE = re.compile(r"ERROR: \w+Sanitizer:\s*([\w-]+)")
_UBSAN_RE = re.compile(r"^([\w./-]+):(\d+):\d+:\s*runtime error:\s*([^\n]+)", re.M)

# absl CHECK / QCHECK / DCHECK / TF_RET_CHECK:
#   F0000 00:00:00 file.cc:123] Check failed: a == b (1 vs. 2)
#   RET_CHECK failure (file.cc:45) cond
_CHECK_RE = re.compile(r"([\w./\-]+\.(?:cc|h|cpp)):(\d+)\]\s*Check failed:\s*([^\n]*)")
_RET_CHECK_RE = re.compile(r"RET_CHECK failure \(([\w./\-]+):(\d+)\)\s*([^\n]*)")
# glibc assert: "hlo-opt: file.cc:12: ret fn(args): Assertion `cond' failed."
_ASSERT_RE = re.compile(r"([\w./\-]+\.(?:cc|h|cpp|inc)):(\d+): [^\n]*?Assertion `([^']*)' failed")
_UNREACHABLE_RE = re.compile(r"UNREACHABLE executed at ([\w./\-]+):(\d+)!?(?::\s*([^\n]*))?")
_LLVM_ERROR_RE = re.compile(r"LLVM ERROR:\s*([^\n]+)")
_FATAL_RE = re.compile(r"^F\d{4} [\d:. ]+\s+\d+\s+([\w./\-]+):(\d+)\]\s*([^\n]*)", re.M)
_SIGNAL_RE = re.compile(
    r"\*\*\* (SIGSEGV|SIGABRT|SIGILL|SIGFPE|SIGBUS)|"
    r"^(Segmentation fault|Illegal instruction|Aborted|Bus error|Floating point exception)"
    r"(?:\s*\(core dumped\))?\s*$", re.M)
# An absl status printed by the tools: `INTERNAL: message` (run_hlo_module
# streams the status; hlo-opt logs "Failed to ...: INTERNAL: ...").
_STATUS_RE = re.compile(r"\b(INTERNAL|UNIMPLEMENTED|INVALID_ARGUMENT|NOT_FOUND|"
                        r"FAILED_PRECONDITION|UNKNOWN|OUT_OF_RANGE|RESOURCE_EXHAUSTED|"
                        r"DEADLINE_EXCEEDED|ABORTED|UNAVAILABLE):\s*([^\n]*)")
_MISMATCH_RE = re.compile(r"first mismatch at array index|"
                          r"Mismatches in |"
                          r"Expected literal to be |"
                          r"mismatch\(es\)|"
                          r"expected value: .*\n\s*actual value: ")
_RUNS_FAILED_RE = re.compile(r"(\d+)/(\d+) runs failed")
# run_hlo_module echoes the module it ran; a random-number op in it makes
# a CPU-vs-interpreter difference expected, not a finding.
_RNG_RE = re.compile(r"\b(rng|rng-bit-generator|rng-get-and-update-state)\(")
_INTERNAL_BUG_RE = re.compile(r"should never|Should never|should not happen|internal error|"
                              r"Internal error|inconsistent|corrupt|Unreachable|unreachable|"
                              r"Failed to compile|codegen|Codegen|emit")
_USAGE_CHECK_RE = re.compile(r"argc|Must specify|Usage:|unknown flag|Unknown command line flag")
_VERIFIER_CONTEXT_RE = re.compile(r"during context \[hlo verifier\]|\[hlo verifier\]|"
                                  r"HloVerifier|Expected instruction to have shape|"
                                  r"was not found|is not a valid|does not match|"
                                  r"parse error|Error parsing|Expected (?:'|\w+ but)")
_CLOSE_ENOUGH_RE = re.compile(r"are close enough")
# Stack frames from the tsl failure signal handler / absl symbolizer:
#   "    @     0x55d4a1c2b4e3  xla::HloInstruction::..." or "#3 0x... in xla::..."
_FRAME_RE = re.compile(r"^\s*(?:@\s+0x[0-9a-f]+\s+|#\d+\s+0x[0-9a-f]+\s+in\s+)([^\n]+?)\s*$", re.M)
_XLA_FRAME_RE = re.compile(r"\b((?:xla|llvm|mlir|tsl)::[\w:<>~ ,*&()\-]+)")

_VOLATILE = [
    (re.compile(r"0x[0-9a-fA-F]{4,}"), "0xADDR"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"\b\d+ vs\. \d+\b"), "N vs. N"),
    (re.compile(r"\b\d{3,}\b"), "N"),
    # messages that quote an instruction or computation name: one class
    (re.compile(r"instruction %?[\w.\-]+ is live"), "instruction <name> is live"),
    (re.compile(r"Instruction %?[\w.\-]+ must have"), "Instruction <name> must have"),
    (re.compile(r"HloInstruction '[^']*'"), "HloInstruction '<name>'"),
    (re.compile(r"evaluated value for: %[^\n]*"), "evaluated value for: <instr>"),
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return text.strip()


def _short(path):
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def is_resource_exhaustion(output):
    return bool(_RESOURCE_RE.search(output or ""))


def _first_xla_frame(output):
    for fm in _FRAME_RE.finditer(output):
        frame = fm.group(1)
        m = _XLA_FRAME_RE.search(frame)
        if m and not re.match(r"(tsl|absl)::.*(Fail|Log|Signal|Check|Abort)", m.group(1)):
            return re.sub(r"\(.*$", "", m.group(1))[:100]
    return None


def classify(output, tool="run_hlo_module", return_code=None):
    """Decide what `output` (stdout+stderr of one run) is.

    Returns a dict:
      kind       one of "crash", "sanitizer", "internal", "mismatch",
                 "timeout", "resource", "rejected", "unsupported", "ok"
      signature  a stable name for the finding, None for non-findings
      is_bug     kind in (crash, sanitizer, internal, mismatch)
      is_valid   the module was accepted (compiled, and ran when asked)
    """
    out = output or ""

    def hit(kind, signature, valid=False):
        if tool == "hlo-opt-passes":
            # hlo-opt --passes runs the named passes on the *unverified*
            # module (there is no verifier in its pass list), so a crash
            # there may be reachable only through HLO the verifier would
            # reject; the tag keeps that visible at triage time.
            signature = f"{signature} [passes]"
        elif tool == "hlo-opt-gpu" and signature:
            # the GPU backend has its own emitters and passes; keep its
            # crashes apart from the CPU ones with the same file:line shape
            signature = f"{signature} [gpu]"
        return {"kind": kind, "signature": signature, "is_bug": True, "is_valid": valid}

    if "TIMEOUT" == out.strip():
        return {"kind": "timeout", "signature": None, "is_bug": False, "is_valid": False}
    if is_resource_exhaustion(out):
        return {"kind": "resource", "signature": None, "is_bug": False, "is_valid": False}

    # 1. Sanitizer reports name the defect precisely.
    m = _ASAN_RE.search(out)
    if m:
        kind = _ASAN_KIND_RE.search(out)
        frame = _first_xla_frame(out)
        where = frame or _normalize(m.group(2))[:80]
        return hit("sanitizer", f"{m.group(1)}: {kind.group(1) if kind else 'error'} in {where}")
    m = _UBSAN_RE.search(out)
    if m:
        return hit("sanitizer", f"UBSan: {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:60]}")

    # 2. XLA's own invariant checks. (A tool's argument-usage CHECK —
    #    `Check failed: argc == 2 Must specify a single input file` — is a
    #    harness mistake, not a finding.)
    m = _CHECK_RE.search(out)
    if m and _USAGE_CHECK_RE.search(m.group(3)):
        return {"kind": "usage", "signature": None, "is_bug": False, "is_valid": False}
    if m and re.search(r"is OK \((INVALID_ARGUMENT|UNIMPLEMENTED|NOT_FOUND|FAILED_PRECONDITION)", m.group(3)):
        # hlo-opt's own CHECK_OK around the pipeline: the wrapped status is
        # the verifier or a pass declining the module, not a crash.
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    if m:
        return hit("crash", f"CHECK {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _RET_CHECK_RE.search(out)
    if m and m.group(1).endswith("hlo_verifier.cc"):
        # The verifier reports through RET_CHECK too; that is a rejection.
        return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    if m:
        return hit("crash", f"RET_CHECK {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _ASSERT_RE.search(out)
    if m:
        return hit("crash", f"Assertion {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _UNREACHABLE_RE.search(out)
    if m:
        return hit("crash", f"UNREACHABLE {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3) or '')[:50]}")
    m = _LLVM_ERROR_RE.search(out)
    if m:
        return hit("crash", f"LLVM ERROR: {_normalize(m.group(1))[:90]}")
    m = _FATAL_RE.search(out)
    if m:
        return hit("crash", f"FATAL {_short(m.group(1))}:{m.group(2)} {_normalize(m.group(3))[:70]}")

    # 3. A signal without any of the above (a plain segfault in the
    #    generated code or the runtime).
    m = _SIGNAL_RE.search(out)
    if m:
        sig = m.group(1) or m.group(2)
        frame = _first_xla_frame(out)
        return hit("crash", f"{sig} in {frame}" if frame else f"{sig} ({tool})")

    # 4. Differential: the CPU executable and the evaluator disagree. The
    #    comparison reports through a status too, so it is looked for
    #    before the status codes are read: "N/M runs failed" alone is not
    #    enough (a reference run that hit UNIMPLEMENTED also fails).
    statuses = _STATUS_RE.findall(out)
    if _MISMATCH_RE.search(out) and not _RNG_RE.search(out):
        mm = re.search(r"first mismatch at array index ([^\n:]*)", out)
        where = ("index " + mm.group(1).strip()) if mm else "result"
        return hit("mismatch", f"Mismatch: CPU vs interpreter ({where})", valid=True)
    # 5. Statuses. INTERNAL is the compiler's own admission; the rest are
    #    the expected fate of an ill-formed module. One exception, measured
    #    on the real binary: the pass pipeline wraps the HLO verifier's
    #    rejection of an ill-typed module as `INTERNAL: during context [hlo
    #    verifier]: Expected instruction to have shape ...` — that is the
    #    verifier doing its job on a fused module, not a compiler fault.
    for code, msg in statuses:
        if code == "INTERNAL":
            # Measured on the first batch (431 bundles): 267 were the PJRT
            # runner declining nested tuples, 24 a pass's own shape check
            # ("during context [Unknown]"), 8 layout assignment declining a
            # bitcast, 11 a GPU backend_config it cannot parse — XLA phrases
            # "I decline this module" as INTERNAL as readily as "I broke".
            # A real internal fault comes with a CHECK / RET_CHECK / signal,
            # which the rules above already catch; a bare INTERNAL is a
            # rejection unless it says so itself.
            if _INTERNAL_BUG_RE.search(msg) and not _VERIFIER_CONTEXT_RE.search(out):
                return hit("internal", f"INTERNAL: {_normalize(msg)[:100]}")
            return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    for code, msg in statuses:
        if code in ("UNIMPLEMENTED", "NOT_FOUND"):
            return {"kind": "unsupported", "signature": None, "is_bug": False, "is_valid": False}
        if code in ("INVALID_ARGUMENT", "FAILED_PRECONDITION", "OUT_OF_RANGE", "UNKNOWN"):
            return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
    if return_code not in (None, 0):
        if re.search(r"error:|Error|failed", out):
            return {"kind": "rejected", "signature": None, "is_bug": False, "is_valid": False}
        return {"kind": "crash", "signature": f"exit {return_code} ({tool})",
                "is_bug": return_code not in (1, 2), "is_valid": False}
    return {"kind": "ok", "signature": None, "is_bug": False, "is_valid": True}


def crash_signature(output, tool="run_hlo_module", return_code=None):
    return classify(output, tool, return_code)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# Opcodes the single-host CPU runner cannot execute or the evaluator does
# not implement; a seed built around them can still be *compiled* by
# hlo-opt, so the driver keeps them on that tool.
COLLECTIVE_OPS = frozenset((
    "all-reduce", "all-gather", "all-to-all", "reduce-scatter",
    "collective-permute", "collective-broadcast", "all-reduce-start",
    "all-reduce-done", "all-gather-start", "all-gather-done",
    "collective-permute-start", "collective-permute-done", "send", "recv",
    "send-done", "recv-done", "partition-id", "replica-id", "ragged-all-to-all",
))
BACKEND_SPECIFIC_OPS = frozenset(("custom-call",))


def analyze_seed(content, filename=""):
    """Facts about one HLO module. Never raises: a module the model cannot
    read still gets a (mostly empty) record so it is kept as a seed."""
    facts = {
        "computations": [], "entry": None, "entry_params": [],
        "entry_root_shape": None, "opcodes": [], "has_collectives": False,
        "has_custom_call": False, "is_scheduled": False,
        "variables": [], "dataflows": [], "line_count": content.count("\n") + 1,
    }
    try:
        mod = parse_module(content)
    except Exception:
        mod = None
    if not mod or not mod.computations:
        return facts
    facts["computations"] = [c.name for c in mod.computations]
    facts["is_scheduled"] = "is_scheduled=true" in mod.header_attrs
    entry = mod.entry()
    opcodes = set()
    for c in mod.computations:
        for ins in c.instrs:
            opcodes.add(ins.opcode)
    facts["opcodes"] = sorted(opcodes)
    facts["has_collectives"] = bool(opcodes & COLLECTIVE_OPS)
    facts["has_custom_call"] = bool(opcodes & BACKEND_SPECIFIC_OPS)
    if entry is not None:
        facts["entry"] = entry.name
        facts["entry_params"] = [{"name": p.name, "shape": normalize_shape(p.shape),
                                  "index": p.param_index()} for p in entry.params()]
        root = entry.root()
        facts["entry_root_shape"] = normalize_shape(root.shape) if root else None
        # Name pool + def-use edges for the generic dataflow machinery.
        names = [ins.name for ins in entry.instrs]
        facts["variables"] = names
        edges = []
        for ins in entry.instrs:
            for src in ins.operand_names():
                if src in names:
                    edges.append([src, ins.name])
        facts["dataflows"] = edges[:400]
    return facts
