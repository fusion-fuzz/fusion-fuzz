"""
projects/lean/analyzer.py — decide what a `lean` run means.

Lean reports its own failures in a handful of distinct ways, and each one
has an innocent twin that a test program can produce on purpose. The
oracle below is the list of the guilty ones, most specific first:

  sanitizer        `SUMMARY: AddressSanitizer: ...`, `runtime error:` from
                   UBSan — the asan toolchain (setup.py) only.
  assert           "LEAN ASSERTION VIOLATION / File: / Line: / <cond>" — a
                   `lean_assert` in the kernel, the runtime or lean.h,
                   live in both from-source toolchains (LEAN_DEBUG).
  panic            "PANIC at <decl> <module>:<line>:<col>: <msg>" raised in
                   one of Lean's own modules. A panic in Init/Std that the
                   *program* triggered (`#[][0]!`, `none.get!`) is not one:
                   it is told apart by the message and by the backtrace
                   Lean prints after every panic — a program's own panic
                   runs under the IR interpreter. A panic in the program's
                   own module is never one. A Lean-internal panic on a seed
                   that does metaprogramming is still reported, labelled
                   `[meta]`: the program may have misused the API.
  internal_panic   "INTERNAL PANIC: ..." from the runtime, except executed
                   `sorry` (the program's) and out of memory (resource).
  kernel           "(kernel) ..." — the kernel rejected a declaration the
                   elaborator produced, i.e. the elaborator built an
                   ill-typed term. Not counted: the kernel's resource limits,
                   and the inductive-declaration checks (positivity,
                   universe levels, nested-inductive shape), which are the
                   kernel's *job* for a user's ill-formed `inductive`.
                   On a seed that calls `addDecl` itself: labelled `[meta]`.
  internal_error   the elaborator's and compiler's own consistency checks:
                   "internal error", the IR checker, the LCNF checker
                   (`compiler.check`), the IR interpreter's "unknown
                   declaration", and "unknown free variable/metavariable"
                   naming an internal `_uniq` id.
  miscompile       (driver "diff" tool) the natively compiled program and
                   the interpreter disagree on output or exit status.
  signal           the process died on a signal with none of the above —
                   Lean's own stack-overflow handler excepted (resource).

Everything else with a non-zero exit is an ordinary rejection.
"""

import re

_VOLATILE = [
    (re.compile(r"/tmp/\S+|\S*\.fused/\S+"), "<tmp>"),
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"_uniq\.\d+"), "_uniq.N"),
    (re.compile(r"\?(?:m|u|_)\.\d+|\?_mvar\.\d+"), "?m.N"),
    (re.compile(r"_hyg\.\d+"), "_hyg.N"),
    (re.compile(r"\b\d{3,}\b"), "N"),
    (re.compile(r"[`'«][^`'»\n]{1,80}[`'»]"), "'_'"),
]


def _normalize(text):
    text = text.strip()
    for rx, rep in _VOLATILE:
        text = rx.sub(rep, text)
    return " ".join(text.split())


_RESOURCE_RE = re.compile(
    r"INTERNAL PANIC: out of memory|std::bad_alloc|Stack overflow detected|"
    r"\(kernel\) deep recursion|\(kernel\) deterministic timeout|"
    r"\(kernel\) excessive memory|\(kernel\) interrupted|maximum recursion depth has been reached|"
    r"AddressSanitizer: stack-overflow|AddressSanitizer: (?:allocation-size-too-big|"
    r"requested allocation size|out of memory|calloc-overflow)|"
    r"AddressSanitizer failed to allocate|Cannot allocate memory|failed to allocate|"
    r"failed to create thread|Resource temporarily unavailable|excessive memory consumption|"
    r"(?:deterministic )?timeout at `[^`]*`, maximum number of heartbeats",
    re.I)

_SAN_RE = re.compile(r"SUMMARY: (\w+Sanitizer): (\S+)(?: ([^\n]+))?")
_UBSAN_RE = re.compile(r"^([^\s:]+\.(?:cpp|c|h)):(\d+):\d+: runtime error: ([^\n]+)", re.M)
_ASSERT_RE = re.compile(r"LEAN ASSERTION VIOLATION\s*\nFile: ([^\n]+)\nLine: (\d+)\n([^\n]*)")
_PANIC_RE = re.compile(r"PANIC at (\S+) ([\w.«»]+):(\d+):(\d+): ([^\n]*)")
_PANIC_NODECL_RE = re.compile(r"PANIC at ([\w.«»]+):(\d+):(\d+): ([^\n]*)")
_INTERNAL_PANIC_RE = re.compile(r"INTERNAL PANIC: ([^\n]+)")
_KERNEL_RE = re.compile(r"\(kernel\) ([^\n]+)")
_KERNEL_USER_RE = re.compile(
    r"non[- ]positive occurrence|invalid nested inductive|universe level of type_of|"
    r"invalid return type for|mutually inductive types must|parameters of all inductive|"
    r"number of parameters mismatch|duplicate constructor|invalid inductive datatype|"
    r"invalid empty \(mutual\) inductive|arg #\d+ of|invalid occurrence of datatype|"
    r"must be applied to the parameters|uses unsafe declaration|"
    r"unknown constant|constant has already been declared|"
    r"declaration has metavariables '[^']*sorry")
_KERNEL_RESOURCE_RE = re.compile(r"deep recursion|deterministic timeout|excessive memory|interrupted")
_INTERNAL_ERROR_RE = re.compile(
    r"(?:^|\W)(internal error[^\n]*|Internal error[^\n]*|"
    r"compiler IR check failed at[^\n]*|"
    r"\(interpreter\) (?:unknown declaration|IR of declaration)[^\n]*|"
    r"invalid LCNF[^\n]*|LCNF (?:let declaration|local function declaration|parameter) mismatch[^\n]*|"
    r"type mismatch at LCNF application[^\n]*|invalid out of scope free variable[^\n]*|"
    r"invalid jump to out of scope join point[^\n]*|free variables are not unique[^\n]*|"
    r"unknown free variable [`']?(?:_fvar|_uniq)[^\n]*|"
    r"unknown metavariable '\?_uniq[^\n]*|unknown (?:universe )?metavariable `\?_uniq[^\n]*|"
    r"unexpected bound variable[^\n]*|loose bound variable[^\n]*)")
# `--run` on a file without `main` (a rename or a reduction removed it).
_IR_USER_RE = re.compile(r"which has no executable code|consider marking definition as 'noncomputable'|"
                         r"\(interpreter\) unknown declaration 'main'")
_SIGNAL_TEXT_RE = re.compile(r"^(Segmentation fault|Aborted|Illegal instruction|Bus error|"
                             r"Floating point exception|Trace/breakpoint trap)", re.M)
_SIGNAL_NAMES = {4: "SIGILL", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE", 11: "SIGSEGV", 5: "SIGTRAP"}
_LEAN_ROOTS = ("Lean", "Init", "Std", "Lake")
# Panics a program raises through the public API of Init/Std.
_USER_PANIC_MSG_RE = re.compile(
    r"index out of bounds|value is none|empty list|head of empty|tail of empty|"
    r"unreachable code|not implemented|Option\.get!|get! |division by zero|"
    r"String\.(?:toNat|toInt)!|invalid|out of range|key is not in the map|"
    r"element not found|is not a valid|empty array|empty string|"
    r"back! |pop!|findIdx|cannot parse|getElem!|must be positive|expected", re.I)
# Where the program's own code is entered: `#eval` (Lean.evalConst ->
# lean_eval_const), `--run` (lean_run_main), `initialize` (lean_run_init).
# The IR interpreter's own frames are static and print unnamed.
_INTERP_FRAME_RE = re.compile(r"lean_eval_const|Lean\.evalConst|lean_run_main|lean_run_init|"
                              r"lean_io_run_main|ir_interpreter|elabEvalCoreUnsafe|Lean\.Elab\.evalTactic.*Unsafe")
_BACKTRACE_RE = re.compile(r"backtrace:\n((?:.*\n?){1,60})")
_FRAME_NAME_RE = re.compile(r"^\s*(?:\S+?\((.+?)\+0x[0-9a-f]+\)|([A-Za-z_][\w.!?'«»]+(?:\.[\w.!?'«»]+)+))", re.M)
_NOISE_FRAME_RE = re.compile(r"lean_panic|panic_fn|print_backtrace|lean_panic_impl|panicCore|"
                             r"panicWithPos|mkPanicMessage|^_start$|__libc|backtrace", re.I)


def _root(mod):
    return mod.strip("«»").split(".", 1)[0]


def _frames(text):
    m = _BACKTRACE_RE.search(text)
    if not m:
        return []
    out = []
    for a, b in _FRAME_NAME_RE.findall(m.group(1)):
        f = a or b
        if f and not _NOISE_FRAME_RE.search(f):
            out.append(f)
    return out


_PLAIN_ERROR_RE = re.compile(r":\d+:\d+: error(?:\([\w.]+\))?: ")


def _bug(kind, sig, meta=False, after_error=False):
    if meta:
        sig = f"{sig} [meta]"
    if after_error:
        # An ordinary error came first: the failure is on Lean's error-
        # recovery path. Still a bug (the kernel must never see what the
        # elaborator recovered to), but ranked below one on valid input.
        sig = f"{sig} [after error]"
    return {"kind": kind, "signature": sig, "is_bug": True, "is_valid": True}


_ORACLE_LINE_RE = re.compile(r"\(kernel\)|unknown free variable|internal error|Internal error|"
                             r"LCNF|IR check failed|\(interpreter\)|PANIC")


def _after_error(out, pos):
    """Is there an ordinary (non-oracle) error message before `pos`? Plain
    `file:l:c: error:` lines and `--json` lines with severity error both
    count. Lean prints messages on stdout and panics on stderr, and the
    driver puts stdout first, so for a panic this means "the run also
    reported an ordinary error" rather than strictly "before"."""
    pos = out.rfind("\n", 0, pos) + 1        # the oracle's own line is not "before"
    for line in out[:pos].split("\n"):
        if _ORACLE_LINE_RE.search(line):
            continue
        if _PLAIN_ERROR_RE.search(line) or '"severity":"error"' in line:
            return True
    return False


def _ok(kind, valid):
    return {"kind": kind, "signature": None, "is_bug": False, "is_valid": valid}


def _expected(program, phrase):
    """The program text itself carries the message — a test of that very
    diagnostic, not a new failure."""
    return bool(program) and phrase and phrase[:60] in program


def classify(output, return_code=None, program="", uses_meta=False, tool="elab",
             program_unsafe=False):
    """{'kind', 'signature', 'is_bug', 'is_valid'} for one run.

    output: stdout + stderr. program: the source that was run (for the
    expected-message check and the meta/unsafe labels). tool: "elab"
    (lean file), "run" (lean --run: also executes main), "diff" (native vs
    interpreted; the driver reports a mismatch itself)."""
    out = output or ""
    # `--json` prints each message as one JSON line, newlines escaped: the
    # same message must give the same signature either way.
    if '"severity"' in out:
        out = out.replace("\\n", "\n").replace('\\"', '"')

    m = _SAN_RE.search(out)
    if m and not re.search(r"stack-overflow|allocation-size-too-big|out of memory|"
                           r"requested allocation size|calloc-overflow", m.group(2)):
        where = _normalize(m.group(3) or "")[:70]
        return _bug("sanitizer", f"{m.group(1)}: {m.group(2)} {where}".strip())
    m = _UBSAN_RE.search(out)
    if m:
        f = m.group(1).split("/")[-1]
        return _bug("sanitizer", f"UBSan: {f}:{m.group(2)} {_normalize(m.group(3))[:70]}")
    m = _ASSERT_RE.search(out)
    if m:
        f = "/".join(m.group(1).strip().split("/")[-2:])
        # `unsafe` code (`unsafeCast`, `evalConst` at the wrong type) can
        # break the runtime's object model by itself; the kernel's and the
        # elaborator's assertions stay bugs regardless.
        if not (program_unsafe and re.search(r"lean\.h|runtime/", m.group(1))):
            return _bug("assert", f"assert {f}:{m.group(2)} {_normalize(m.group(3))[:80]}")

    for m in _PANIC_RE.finditer(out):
        decl, mod, line, _col, msg = m.groups()
        root = _root(mod)
        if root not in _LEAN_ROOTS:
            continue                     # the program's own `panic!`
        frames = _frames(out[m.start():])
        # Frames between the panic and the point where the program's own
        # code was entered: if that point exists, only these are Lean's.
        k = next((i for i, f in enumerate(frames) if _INTERP_FRAME_RE.search(f)), None)
        interpreted = k is not None
        lean_side = frames[:k] if interpreted else frames
        lean_internal = any(f.startswith(("Lean.Elab", "Lean.Meta", "Lean.Compiler", "Lean.PrettyPrinter",
                                          "Lean.Parser", "Lean.Server", "Lean.Linter", "Lean.IR"))
                            for f in lean_side)
        if root in ("Init", "Std") and (interpreted or _USER_PANIC_MSG_RE.search(msg)) and \
                not lean_internal:
            continue                     # the program's misuse of Init/Std
        if root in ("Init", "Std") and uses_meta and _USER_PANIC_MSG_RE.search(msg):
            # A metaprogram's own extension (a delaborator, an elaborator,
            # a simproc) runs *inside* Lean's frames, interpreted, and the
            # interpreter's frames are unnamed: its `get!` looks like
            # Lean's.
            continue
        if root == "Lean" and interpreted and uses_meta and not lean_internal:
            continue                     # a metaprogram's own call into Lean's API
        if _expected(program, msg):
            continue
        caller = next((f for f in frames if f.startswith("Lean.") and f != decl), "")
        sig = f"panic {decl} @{mod}:{line}: {_normalize(msg)[:70]}"
        if caller and root != "Lean":
            sig += f" < {caller[:60]}"
        return _bug("panic", sig, meta=uses_meta and root == "Lean",
                    after_error=_after_error(out, len(out)))
    for m in _PANIC_NODECL_RE.finditer(out):
        mod, line, _col, msg = m.groups()
        if _root(mod) == "Lean" and not _expected(program, msg):
            return _bug("panic", f"panic @{mod}:{line}: {_normalize(msg)[:70]}", meta=uses_meta,
                        after_error=_after_error(out, len(out)))

    m = _INTERNAL_PANIC_RE.search(out)
    if m:
        msg = m.group(1)
        if "sorry" in msg:
            pass
        elif "out of memory" in msg:
            return _ok("resource", False)
        elif not (program_unsafe and tool == "run"):
            return _bug("internal_panic", f"INTERNAL PANIC: {_normalize(msg)[:80]}")

    # A resource limit hit by one command does not excuse a kernel or
    # internal error reported for another; it only decides what a bare
    # signal or a timeout means.
    resource = bool(_RESOURCE_RE.search(out))

    for m in _KERNEL_RE.finditer(out):
        msg = m.group(1)
        if _KERNEL_USER_RE.search(msg) or _KERNEL_RESOURCE_RE.match(msg) or \
                _expected(program, "(kernel) " + msg):
            continue
        kind_phrase = re.split(r"[,'`]|\bat\b|:\s*$", msg, maxsplit=1)[0].strip()
        # A program that hands terms to the kernel itself (`addDecl`,
        # `Kernel.isDefEq`/`whnf`/`check`) gets the kernel's verdict on
        # *its* terms.
        meta = bool(re.search(r"\baddDecl\b|\bDeclaration\.|\bmkAuxDefinition\b|\bKernel\.\w+|"
                              r"ofExceptKernelException", program or ""))
        return _bug("kernel", f"(kernel) {_normalize(kind_phrase)[:80]}", meta=meta,
                    after_error=_after_error(out, m.start()))

    check_types = bool(re.search(r"compiler\.checkTypes\s+true", program or ""))
    for m in _INTERNAL_ERROR_RE.finditer(out):
        msg = m.group(1)
        if _IR_USER_RE.search(msg) or _expected(program, msg[:50]):
            continue
        if check_types and "LCNF" in msg:
            continue                     # documented as incomplete for dependent types
        if msg.lower().startswith("internal error") and program and \
                re.search(r"throwError\s+m?!?\"[^\"]*internal error", program, re.I):
            continue
        return _bug("internal_error", _normalize(re.sub(r"\s+", " ", msg))[:100],
                    after_error=_after_error(out, m.start()))

    if return_code == 124:
        return _ok("timeout", False)
    if resource:
        return _ok("resource", False)

    sig_no = None
    if return_code is not None:
        if return_code < 0:
            sig_no = -return_code
        elif return_code >= 128 and return_code != 255:
            sig_no = return_code - 128
    m = _SIGNAL_TEXT_RE.search(out)
    if sig_no in _SIGNAL_NAMES or m:
        name = _SIGNAL_NAMES.get(sig_no) or m.group(1)
        if program_unsafe:
            # `#eval` runs the program's own `unsafe` code during
            # elaboration too, not only under `--run`.
            return _ok("rejected", False)
        where = " < ".join(f[:40] for f in _frames(out)[:3])
        return _bug("signal", f"lean {name}" + (f" in {where}" if where else ""))

    if return_code in (0, None):
        return _ok("ok", True)
    return _ok("rejected", False)


def crash_signature(output, return_code=None, program=""):
    return classify(output, return_code, program)["signature"]
