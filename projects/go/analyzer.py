"""
projects/go/analyzer.py — the Go-specific analysis half of the adapter.

Two kinds of analysis, both consumed by projects/go/driver.py:

  * Output analysis (classify / crash_signature / is_resource_exhaustion) —
    decide whether what the toolchain printed is a bug, and what to call it.
  * Seed analysis (analyze_seed) — the facts about a Go source file that
    determine how it must be compiled and how it may be fused.

They live apart from driver.py because they are pure functions of text:
no toolchain, no temp directory, no subprocess, so they can be tested
directly.

The one thing to know about Go's compiler before reading the oracle
---------------------------------------------------------------------
base.FatalfAt (src/cmd/compile/internal/base/print.go) reports an
internal compiler error like this:

    if Debug.Panic != 0 || numErrors == 0 {
        fmt.Printf("%v: internal compiler error: ", ...)
    }

Read the condition carefully: **if the file already produced ordinary
errors, an ICE prints nothing at all** and the compiler quietly exits.
That is reasonable for users — an internal error after a type error is
usually a consequence of it — and ruinous for this fuzzer, because a
fused program is ill-formed far more often than not. Without
`-gcflags=-d=panic` (which sets Debug.Panic) most ICEs this fuzzer
triggers would be invisible, and the run would look like it was simply
not finding anything. driver.py passes that flag on every compilation.

A second consequence: because the toolchain is built from source, its
version string contains "devel", so Go dumps a full stack trace with the
ICE rather than the "please file a bug report" blurb a release build
prints. That stack is what crash_signature groups on.
"""

import re

# ---------------------------------------------------------------------------
# Output analysis
# ---------------------------------------------------------------------------

# Shapes that mean the machine ran out of something. Not bugs: the same
# input on a bigger box compiles, so there is nothing to reduce or report.
_RESOURCE_RE = re.compile(
    r"fatal error: runtime: out of memory"
    r"|runtime: out of memory"
    r"|fatal error: out of memory"
    r"|cannot allocate memory"
    r"|signal: killed"                       # OOM killer or our own timeout
    r"|no space left on device"
    r"|too many open files"
    r"|fatal error: stack overflow",         # see the note in classify()
    re.IGNORECASE)

# "./x.go:9:5: internal compiler error: bad type"
_ICE_RE = re.compile(r"internal compiler error:\s*([^\n]+)")

# A compiler panic that did not go through Fatalf.
#   panic: runtime error: index out of range [3] with length 2
_PANIC_RE = re.compile(r"^panic:\s*([^\n]+)", re.M)

# Runtime fatal errors — "concurrent map writes", "all goroutines are
# asleep - deadlock!". These are compiler bugs too when the compiler is
# what died.
_FATAL_RE = re.compile(r"^fatal error:\s*([^\n]+)", re.M)

# The race detector, when the compiler was built with FFL_GO_RACE=1.
_RACE_RE = re.compile(r"WARNING: DATA RACE")

# Stack frames look like:
#   cmd/compile/internal/ssa.(*Func).Fatalf(0xc0001, ...)
#           /go/src/cmd/compile/internal/ssa/func.go:722 +0x1b8
_FRAME_RE = re.compile(r"^(cmd/compile/[\w/]+\.[\w.()*]+)\(", re.M)

# Frames that are part of *reporting* the failure rather than causing it;
# grouping on these would collapse every ICE into one bucket.
_REPORTING_FRAMES = (
    "runtime/debug.Stack",
    "cmd/compile/internal/base.FatalfAt",
    "cmd/compile/internal/base.Fatalf",
    "cmd/compile/internal/base.Assert",
    "cmd/compile/internal/base.Assertf",
    "cmd/compile/internal/base.AssertfAt",
    "cmd/compile/internal/base.ErrorfAt",
    "cmd/compile/internal/ssa.(*Func).Fatalf",
    "cmd/compile/internal/ssa.(*Value).Fatalf",
    "cmd/compile/internal/ssa.(*Block).Fatalf",
    # Panic plumbing. The compiler converts a panic into an ICE by
    # recovering and re-panicking through several layers, so a real crash
    # arrives with these on top of the frame that actually failed. Grouping
    # on them would collapse every panic-routed ICE into one bucket.
    "cmd/compile/internal/gc.handlePanic",
    "cmd/compile/internal/types2.(*Checker).handleBailout",
    "cmd/compile/internal/types2.(*Checker).objDecl.func",
    "cmd/compile/internal/noder.checkFiles.func",
)

# Detail that varies run to run and would defeat deduplication: pointer
# values, temporary paths, autotmp numbering, goroutine ids.
_VOLATILE = [
    (re.compile(r"0x[0-9a-f]{4,}"), "0xADDR"),
    (re.compile(r"\b\.autotmp_\d+"), ".autotmp_N"),
    (re.compile(r"\bgoroutine \d+"), "goroutine N"),
    (re.compile(r"/tmp/[\w./-]+"), "TMP"),
    (re.compile(r"\bv\d+\b"), "vN"),            # SSA value names
    (re.compile(r"\bb\d+\b"), "bN"),            # SSA block names
]


def _normalize(text):
    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    return text.strip()[:160]


def is_resource_exhaustion(output):
    """True when the toolchain died for lack of memory, disk or time."""
    return bool(_RESOURCE_RE.search(output or ""))


def _failing_frame(output):
    """The innermost compiler frame that is not part of error reporting."""
    for frame in _FRAME_RE.findall(output or ""):
        if not any(frame.startswith(r) for r in _REPORTING_FRAMES):
            return frame
    return None


def classify(output):
    """Categorise the toolchain's combined stdout+stderr.

    Returns a dict with:
        kind       one of "race", "ice", "panic", "fatal", "resource",
                   "diagnostic", "clean"
        signature  stable grouping key, or None when not a finding
        frame      the failing compiler frame, when the output had a stack
        is_bug     whether this should be saved

    Order matters. A resource failure has to be recognised before the
    panic/fatal handling, because Go reports running out of memory through
    exactly the same "fatal error:" channel as a real compiler bug.
    """
    text = output or ""

    def hit(kind, signature):
        return {"kind": kind, "signature": signature,
                "frame": _failing_frame(text), "is_bug": True}

    # A race report names two conflicting accesses inside the compiler.
    # Strictly more informative than whatever crash follows it.
    if _RACE_RE.search(text):
        frame = _failing_frame(text)
        return hit("race", f"RACE: in {frame}" if frame else "RACE: (no compiler frame)")

    # Before anything that reads "fatal error:".
    #
    # Stack overflow is deliberately counted here rather than as a bug:
    # deeply nested types and expressions are exactly what fusion produces,
    # and Go's parser recursing until the goroutine stack limit is a
    # property of the input's depth, not a defect worth filing.
    if is_resource_exhaustion(text):
        return {"kind": "resource", "signature": None,
                "frame": None, "is_bug": False}

    m = _ICE_RE.search(text)
    if m:
        frame = _failing_frame(text)
        detail = _normalize(m.group(1))
        # The frame is the better grouping key when there is one: the same
        # broken invariant reached from two places is two bugs, and the
        # message often embeds the offending type or value.
        return hit("ice", f"ICE: {detail} [{frame}]" if frame else f"ICE: {detail}")

    m = _PANIC_RE.search(text)
    if m:
        frame = _failing_frame(text)
        detail = _normalize(m.group(1))
        return hit("panic", f"PANIC: {detail} [{frame}]" if frame else f"PANIC: {detail}")

    m = _FATAL_RE.search(text)
    if m:
        frame = _failing_frame(text)
        detail = _normalize(m.group(1))
        return hit("fatal", f"FATAL: {detail} [{frame}]" if frame else f"FATAL: {detail}")

    # An ordinary diagnostic: "./x.go:5:2: undefined: foo". The expected
    # outcome for most fused programs, and not interesting.
    if re.search(r"^[^\s:]*\.go:\d+", text, re.M) or "too many errors" in text:
        return {"kind": "diagnostic", "signature": None,
                "frame": None, "is_bug": False}
    return {"kind": "clean", "signature": None, "frame": None, "is_bug": False}


def crash_signature(output):
    """The grouping key for a finding, or None if *output* is not one."""
    return classify(output)["signature"]


# ---------------------------------------------------------------------------
# Seed analysis
# ---------------------------------------------------------------------------

# Go's test files open with the action the test harness should take.
# `errorcheck` files are deliberately ill-formed with the expected
# diagnostics asserted inline, the way clang's -verify tests are.
_DIRECTIVE_RE = re.compile(r"^//\s*(run|compile|build|errorcheck|runoutput|"
                           r"errorcheckoutput|rundir|compiledir|errorcheckdir|"
                           r"skip|runindir)\b", re.M)

_PACKAGE_RE = re.compile(r"^package\s+(\w+)", re.M)

# Both import spellings: a single `import "fmt"` and a parenthesised block.
_IMPORT_ONE_RE = re.compile(r'^import\s+((?:[\w.]+\s+)?(?:"[^"]+"|`[^`]+`))', re.M)
_IMPORT_BLOCK_RE = re.compile(r'^import\s*\(([^)]*)\)', re.M | re.S)
_IMPORT_SPEC_RE = re.compile(r'((?:[\w.]+\s+)?(?:"[^"]+"|`[^`]+`))')

# `//go:build` / `// +build` constrain which platforms a file compiles on.
_BUILD_TAG_RE = re.compile(r"^//(?:go:build|\s*\+build)\s+(.+)$", re.M)

# Top-level declarations, used to spot the collisions fusion creates.
_TOPLEVEL_FUNC_RE = re.compile(r"^func\s+(?:\([^)]*\)\s*)?(\w+)", re.M)
_TOPLEVEL_TYPE_RE = re.compile(r"^type\s+(\w+)", re.M)


def imports_of(content):
    """Every import spec in *content*, block and single forms alike.

    Fusion has to merge two files' imports into one block, and Go makes an
    unused import a compile *error*, so knowing exactly what each side
    imports is not optional.
    """
    specs = []
    for m in _IMPORT_BLOCK_RE.finditer(content or ""):
        specs.extend(s.strip() for s in _IMPORT_SPEC_RE.findall(m.group(1)))
    for m in _IMPORT_ONE_RE.finditer(content or ""):
        specs.append(m.group(1).strip())
    return list(dict.fromkeys(specs))


def analyze_seed(content, filename=""):
    """Facts about a Go seed that fusion and execution both need.

    Returns a dict:
        directive     the test action ("run"/"compile"/"errorcheck"/...)
        package       the package clause's name, or None
        is_main       whether it is `package main` (i.e. can link)
        imports       list of import specs
        build_tags    //go:build constraints, which can make a file compile
                      to nothing on this platform
        funcs, types  top-level names, the ones that collide when fused
        has_cgo       imports "C", which needs a working gcc and cannot be
                      fused with anything sensibly
    """
    text = content or ""
    m = _DIRECTIVE_RE.search(text)
    directive = m.group(1) if m else None

    m = _PACKAGE_RE.search(text)
    package = m.group(1) if m else None

    specs = imports_of(text)
    return {
        "directive": directive,
        "package": package,
        "is_main": package == "main",
        "imports": specs,
        "build_tags": _BUILD_TAG_RE.findall(text),
        "funcs": _TOPLEVEL_FUNC_RE.findall(text),
        "types": _TOPLEVEL_TYPE_RE.findall(text),
        "has_cgo": any(s.strip('"`') == "C" for s in specs),
    }
