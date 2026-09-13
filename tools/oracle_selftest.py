#!/usr/bin/env python3
"""
tools/oracle_selftest.py — check that a project's crash oracle fires on
the outputs it is meant to catch and stays quiet on the ones it is not.

An adapter's oracle is two lines of pattern matching and it decides
everything: a pattern that is too broad floods the run with bundles that
are the program's fault (R's "unimplemented type" was one), and one that
is too narrow silently reports nothing at all for days. Neither shows up
in the validity rate. This pins both directions with recorded outputs.

    python3 tools/oracle_selftest.py [project ...]
    python3 tools/oracle_selftest.py --sweep [project ...]

The first form runs the recorded cases below. `--sweep` instead replays
every saved bundle's own test.out through its project's oracle: a kept
bundle must still be recognised. That is the check a hand-written list
cannot give — the outputs are real, and there are hundreds of them. It
found three things on its first run: an adapter whose driver class this
file had named wrongly (199 lfortran bundles unchecked), twelve bundles
saved with only the tail of their output because core/utils.py had no
crash anchor for that project (fixed there), and two stale bundles kept
by an oracle that used to match crash text echoed back inside a seed's
own comment.

A dismissed bundle that the oracle still reports is printed but does not
fail the run: those were dismissed by judgement — a test that crashes
the target on purpose, an ASan report about host memory — and no pattern
separates them from a real crash.

Exits non-zero if any case disagrees. Add a project by adding its rows
to CASES: (name, text the interpreter/compiler produced, is it a bug).
"""

import importlib.util
import os
import sys

import yaml

# Run from anywhere: the drivers import `core.driver`, which is only
# importable with the repository root on the path.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

CASES = {
    "ruby": ("RubyDriver", [
        ("rb_bug", "t.rb:3: [BUG] Segmentation fault at 0x0000\n"
                   "-- Control frame information ---\nc:0003 p:---- CFUNC :foo\n", True),
        ("ruby_assert", "../array.c:422: Assertion Failed: "
                        "ary_resize_capa:RARRAY_LEN(ary) <= capacity\n", True),
        ("asan", "SUMMARY: AddressSanitizer: heap-use-after-free /x/y.c:1 in rb_ary_push\n", True),
        # Ruby turns a guard-page hit into SystemStackError; under ASan the
        # sanitizer sees it first. Every unbounded recursion would report.
        ("stack_overflow", "SUMMARY: AddressSanitizer: stack-overflow on address 0x7f\n", False),
        # The fused program exhausted the rss cap; Ruby's ASan death callback
        # then prints a [BUG] banner with a full frame dump.
        ("rss_limit", "==1==AddressSanitizer: hard rss limit exhausted (2048Mb vs 2054Mb)\n"
                      "[BUG] ASAN error\n-- Control frame information ---\n", False),
        # Ruby's own allocator, asked for more than the machine has: the
        # GC ran, could still not satisfy the request, and rb_bug reported.
        ("try_with_gc", "t.rb:43: [BUG] TRY_WITH_GC: could not allocate:"
                        "34359345288 bytes for mem = malloc(size)\n"
                        "-- Control frame information ---\nc:0002 p:0012\n", False),
        # A seed that merely prints the banner (test_rubyoptions does).
        ("printed_banner", '[BUG] not really\n', False),
        ("ordinary_error", "t.rb:1:in '<main>': undefined method 'x' for main (NoMethodError)\n", False),
    ]),
    "r": ("RDriver", [
        ("segfault", "\n *** caught segfault ***\naddress 0x8, cause 'memory not mapped'\n"
                     "\nTraceback:\n 1: foo(x)\n", True),
        ("asan", "SUMMARY: AddressSanitizer: heap-buffer-overflow /x/y.c:9 in Rf_allocVector\n", True),
        ("internal_error", "Error: Internal error: invalid SEXPTYPE\n", True),
        ("c_stack", "Error: C stack usage 7970960 is too close to the limit\n", False),
        ("nested_too_deeply", "Error: evaluation nested too deeply: infinite recursion\n", False),
        ("alloc_failure", "Error: cannot allocate vector of size 4.0 Gb\n", False),
        # UNIMPLEMENTED_TYPE is also R's catch-all for a wrong argument.
        ("unimplemented_type", "Error in options(x) : unimplemented type 'language' in 'options'\n", False),
        ("ordinary_error", "Error in f(x) : object 'y' not found\n", False),
    ]),
    "typescript": ("TypeScriptDriver", [
        ("panic", "panic: Debug failure. Unexpected node.\n\ngoroutine 1 [running]:\n"
                  "github.com/microsoft/TypeScript/tsc/internal/checker.(*Checker).checkX(0x1)\n", True),
        ("go_fatal", "fatal error: concurrent map writes\n\ngoroutine 5:\n", True),
        # Deep user recursion in the checker; core.ApplyDebugStackLimit exists
        # precisely because the limit is expected to be reached.
        ("stack_overflow", "fatal error: stack overflow\n\nruntime stack:\n", False),
        ("oom", "fatal error: out of memory\n", False),
        ("type_error", "t.ts(1,7): error TS2322: Type 'number' is not assignable to type 'string'.\n", False),
    ]),
    # CPython has produced no bundle in the whole campaign, so the bundle
    # sweep cannot check its oracle at all — these recorded cases are the
    # only thing standing between "CPython is robust" and "the oracle
    # never fires". Every branch of projects/cpython/analyzer.classify is
    # represented in both directions.
    "cpython": ("CPythonDriver", [
        ("asan", "SUMMARY: AddressSanitizer: heap-use-after-free "
                 "/src/Objects/listobject.c:230 in list_ass_slice\n", True),
        ("ubsan", "Objects/longobject.c:1204:9: runtime error: signed integer "
                  "overflow: 9223372036854775807 + 1 cannot be represented in type 'long'\n", True),
        ("fatal", "Fatal Python error: _PyObject_GC_UNTRACK: object already untracked\n"
                  "Current thread 0x00007f (most recent call first):\n", True),
        ("assert", "python: Objects/dictobject.c:1523: insertdict: "
                   "Assertion `PyDict_Check(mp)' failed.\n", True),
        ("debug_alloc", "Debug memory block at address p=0x5555: API 'o'\n"
                        "    bad trailing pad byte at tail+0\n", True),
        ("c_api", "SystemError: <built-in function foo> returned NULL "
                  "without setting an exception\n", True),
        ("signal", "Segmentation fault (core dumped)\n", True),
        # Not findings: the driver's own memory cap, ordinary exceptions,
        # and a program that merely names an exception class.
        ("recursion", "RecursionError: maximum recursion depth exceeded\n", False),
        ("memory_error", "MemoryError\n", False),
        ("ordinary_exception", "Traceback (most recent call last):\n"
                               "  File \"t.py\", line 1, in <module>\n"
                               "TypeError: unsupported operand type(s)\n", False),
        ("named_in_string", "the string 'ConnectionAbortedError' was printed\n", False),
        ("clean", "42\n", False),
    ]),
    "julia": ("JuliaDriver", [
        ("internal_error", "Internal error: encountered unexpected error in runtime:\n"
                           "MethodError(f=typeof(convert)())\nStacktrace:\n"
                           " [1] jl_type_error at /src/rtutils.c:129\n", True),
        ("assert_build", "julia: /src/julia/src/gc.c:1201: gc_mark_outrefs: "
                         "Assertion failed: obj is not a valid object\n", True),
        # The glibc form, which is what an assert build actually prints.
        # It carries no "Assertion failed:" text, so it used to fall
        # through to the generic signature and come out as "unknown".
        ("c_assert", "julia: /src/julia/src/genericmemory.c:412: jl_memoryrefunset: "
                     "Assertion `(char*)m.ptr_or_offset - (char*)m.mem->ptr < x' failed.\n"
                     "\n[114284] signal 6 (-6): Aborted\n", True),
        ("llvm_error", "LLVM ERROR: Cannot select: 0x55 v4i64 = bitcast\n", True),
        ("signal", "\nsignal (11): Segmentation fault\nin expression starting at t.jl:3\n"
                   "jl_apply_generic at /src/julia/src/gf.c:3077\n", True),
        # Unbounded recursion, which a fused program reaches constantly:
        # Julia detects the guard page and throws.
        ("stack_overflow", "ERROR: LoadError: StackOverflowError:\nStacktrace:\n [1] f(x)\n", False),
        ("alloc_failure", "ERROR: LoadError: OutOfMemoryError()\nStacktrace:\n [1] Array\n", False),
        # A test that prints the words; a real report carries a backtrace.
        ("printed_words", 'ERROR: LoadError: "Internal error: nope"\n', False),
        ("ordinary_error", "ERROR: LoadError: MethodError: no method matching f(::Int64)\n"
                           "Stacktrace:\n [1] top-level scope\n", False),
    ]),
}


# Every project's driver class, for the bundle sweep. The recorded cases
# above cover the three newest adapters in both directions; the sweep
# below covers all sixteen against real saved output.
DRIVER_CLASSES = {
    "clang": "ClangDriver", "gcc": "GCCDriver", "flang": "FlangDriver",
    "lfortran": "LfortranDriver", "cpython": "CPythonDriver", "go": "GoDriver",
    "rust": "RustDriver", "php": "PHPDriver", "swift": "SwiftDriver",
    "haskell": "HaskellDriver", "mlir": "MLIRDriver", "naga": "NagaDriver",
    "tint": "TintDriver", "ruby": "RubyDriver", "r": "RDriver",
    "typescript": "TypeScriptDriver", "julia": "JuliaDriver",
}


def driver_class_of(project):
    """The driver class a project's driver.py defines."""
    if project in DRIVER_CLASSES:
        return DRIVER_CLASSES[project]
    path = os.path.join(_ROOT, "projects", project, "driver.py")
    if not os.path.exists(path):
        return None
    found = re.findall(r'^class (\w*Driver)\(', open(path).read(), re.M)
    return found[-1] if found else None


def sweep_bundles(projects):
    """Check every saved bundle's own output against its project's
    oracle: a kept bundle must still be recognised as a crash, and a
    dismissed one must stay quiet.

    This is the part a hand-written case list cannot cover. The outputs
    are real — they are what the fuzzer actually saw — and a driver
    change that narrows the oracle shows up here as a bundle the project
    would no longer report, which is otherwise invisible until a week of
    fuzzing comes back empty.
    """
    failures = 0
    for project in projects:
        cls_name = driver_class_of(project)
        base = os.path.join(_ROOT, "output", "bugs", project)
        if not cls_name or not os.path.isdir(base):
            continue
        try:
            driver = load_driver(project, cls_name)
        except Exception as e:
            print(f"  {project:11s} cannot load driver ({e})")
            failures += 1
            continue
        kept = missed = dismissed_ok = dismissed_bad = truncated = 0
        for expected, root in ((True, base), (False, os.path.join(base, "_dismissed"))):
            if not os.path.isdir(root):
                continue
            for name in sorted(os.listdir(root)):
                # Only the `_dismissed` subdirectory is not a bundle. An
                # earlier `startswith("_")` skipped every bundle whose
                # signature begins with one, which is how ruby's
                # `_BUG__ASAN_error...` — a kept bundle and three
                # dismissed ones — went unchecked.
                if name == "_dismissed":
                    continue
                out = os.path.join(root, name, "test.out")
                if not os.path.isfile(out):
                    continue
                text = open(out, encoding="utf-8", errors="replace").read()
                got = driver._check_crash("", text, 1)
                if expected:
                    kept += bool(got)
                    if not got:
                        # A bundle saved before core/utils.py learned this
                        # project's crash anchor kept only the tail of the
                        # output, so the line the oracle needs is not in
                        # the file. That is a defect in the *bundle*, not
                        # in the oracle, and it is counted separately.
                        if "truncated" in text[:400]:
                            truncated += 1
                            continue
                        missed += 1
                        if missed <= 3:
                            print(f"  {project}: MISSED {name[:58]}")
                else:
                    dismissed_ok += not got
                    dismissed_bad += bool(got)
                    if got and dismissed_bad <= 3:
                        print(f"  {project}: dismissed bundle still reported: {name[:48]}")
        # A dismissed bundle the oracle still reports is not a failure of
        # the oracle: those were dismissed by judgement (a test that
        # crashes the target on purpose, a host-memory ASan report), and
        # no pattern separates them from a real one. They are counted so
        # the number is visible, not so it fails the run.
        failures += missed
        line = f"  {project:11s} {kept:3d} recognised, {missed:2d} missed"
        if truncated:
            line += f", {truncated:2d} saved without the crash line"
        if dismissed_ok or dismissed_bad:
            line += f"; dismissed: {dismissed_ok:2d} quiet, {dismissed_bad:2d} reported"
        print(line)
    return failures


def load_driver(project, cls_name):
    spec = importlib.util.spec_from_file_location(
        f"ffl_{project}_driver", os.path.join(_ROOT, "projects", project, "driver.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    config = yaml.safe_load(open(os.path.join(_ROOT, "projects", project, "config.yaml")))
    return getattr(mod, cls_name)(config)


def main(argv):
    sweep = "--sweep" in argv
    argv = [a for a in argv if a != "--sweep"]
    projects = argv[1:] or (sorted(DRIVER_CLASSES) if sweep else sorted(CASES))
    failures = 0
    if sweep:
        print("== bundle sweep: every saved output through its own oracle")
        failures = sweep_bundles(projects)
        print("all bundles agree" if not failures else f"{failures} bundle(s) disagree")
        return 1 if failures else 0
    for project in projects:
        if project not in CASES:
            print(f"{project}: no recorded cases")
            continue
        cls_name, cases = CASES[project]
        driver = load_driver(project, cls_name)
        print(f"== {project}")
        for name, text, expected in cases:
            got = driver._check_crash("", text, 1)
            sig = driver.extract_crash_signature("", text, 1) if got else ""
            ok = got == expected
            failures += 0 if ok else 1
            mark = "ok  " if ok else "FAIL"
            extra = f"  sig={sig[:70]}" if got else ""
            print(f"  {mark} {name:20s} crash={got}{extra}")
    print("all cases agree" if not failures else f"{failures} case(s) disagree")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
