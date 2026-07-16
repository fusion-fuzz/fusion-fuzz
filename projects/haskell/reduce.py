import glob
import os
import shutil
import subprocess
import sys

# Allow running this file standalone (e.g. `python3 projects/haskell/reduce.py`
# from anywhere) as well as via the framework, which already has the repo
# root on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from projects.haskell.setup import _COMPILE_ERROR_MARKERS

stdouterr = None


def _resolve_ghc():
    """Locate ghc the same way projects/haskell/driver.py does."""
    found = shutil.which("ghc")
    if found:
        return found
    matches = glob.glob("/opt/ghc/*/bin/ghc")
    return sorted(matches)[-1] if matches else "ghc"


def _load_crash_patterns():
    """Read analysis.crash_patterns from config.yaml so the default
    bug_output marker stays in sync with what the driver actually treats
    as a crash."""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        patterns = config.get("analysis", {}).get("crash_patterns", [])
        return patterns or ["panic!"]
    except Exception:
        return ["panic!"]


# Function to run the test command and check for bug presence
def run_test(cmd, bug_output):
    """
    Executes the provided `ghc -fno-code` command and checks if the
    expected crash marker (a GHC panic / internal-error string) appears
    in the stdout/stderr. The driver never links or runs the compiled
    program, so every reproducer is a compile-time crash.

    GHC's diagnostics echo the erroneous source line verbatim (e.g. "In
    the expression: ..."), so once delta debugging breaks a line's layout
    badly enough to produce a *compile* error, a bug_output marker that
    happens to be substring of the source itself (very common — crash
    markers are frequently drawn from a user's own `error "..."` call)
    would otherwise look like a match despite the runtime crash no longer
    occurring at all. Reject any result that looks like a compile-time
    failure unless bug_output is itself a compile-diagnostic string.
    """
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=15,
        )
    except Exception:
        return False

    combined = result.stdout + result.stderr
    if bug_output not in _COMPILE_ERROR_MARKERS and any(m in combined for m in _COMPILE_ERROR_MARKERS):
        return False

    found = bug_output in result.stdout or bug_output in result.stderr
    if found:
        global stdouterr
        if stdouterr is None:
            stdouterr = result.stdout + result.stderr

    return found


# Function to minimize the test case by removing lines
def minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    print("reducing .. it may cost some times")
    """
    Minimizes the test case by iteratively removing lines and checking
    if the bug still reproduces. Uses a stepwise approach for efficiency.

    Line-based removal can produce layout-invalid Haskell (GHC's
    off-side rule is whitespace-sensitive), but that's harmless here:
    run_test() only accepts a candidate if the crash still reproduces, so
    a reduction that breaks layout (and therefore no longer crashes with
    the expected marker) is simply rejected and the search continues.
    """
    n = len(lines)
    step = max(n // 2, 1)  # Start with removing half of the lines at a time

    init_step = step

    # Reduce the number of lines step by step
    while step > 0:
        print(f"Current step: {step}")

        # Try removing 'step' lines at a time
        for i in range(0, n, step):
            temp_lines = lines[:i] + lines[i+step:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))

            # If the bug reproduces, accept this as the minimized version
            if run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
        else:
            step //= 2  # If no further reduction is found, reduce step size

    return lines, init_step


# Function for further minimizing by removing multiple lines at a time
def further_minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    """
    Further minimizes the test case by removing 2 to 5 lines at a time
    and checking if the bug still reproduces.
    """
    n = len(lines)

    # Try removing 2 to 5 lines at a time
    for count in range(2, 6):
        # Try removing 'count' lines from each part of the test case
        for i in range(n - count + 1):
            temp_lines = lines[:i] + lines[i+count:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))

            # If the bug reproduces, accept this as the minimized version
            if run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break

    return lines


def minimize_ghc_flags(testpath, ghc_bin, ghc_flags, bug_output):
    """
    Drops extra -O.../-X... flags one at a time while the crash still
    reproduces — the Haskell analogue of PHP's -d ini-flag reduction,
    since ghc_flags are what projects/haskell/driver.py randomizes per run
    (see HaskellDriver.GHC_FLAG_SETS). -fno-code -v0 are always kept
    (that's the fixed, non-negotiable invocation the driver itself uses).
    """
    reduced = list(ghc_flags)
    while True:
        found_shorter = False
        for i in range(len(reduced)):
            candidate = reduced[:i] + reduced[i + 1:]
            cmd = " ".join([ghc_bin, *candidate, testpath])
            if run_test(cmd, bug_output) or run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = candidate
                found_shorter = True
                break
        if not found_shorter:
            break
    return reduced


def reduce_haskell(testpath, ghc_bin, ghc_flags, bug_output):
    """
    Minimizes a Haskell crash reproducer:
      1. Confirms `ghc -fno-code` on the reproducer actually triggers
         bug_output (the driver never links or runs the program, so this
         is always a compile-time crash).
      2. Delta-debugs the .hs source, line by line, in-place at testpath.
      3. Drops unnecessary extra flags while the crash still holds.
    Returns (minimized_source, minimized_ghc_flags).
    """
    reproduce_cmd = " ".join([ghc_bin, *ghc_flags, testpath])

    # Initial test to verify if the reproduce command triggers the bug
    if not run_test(reproduce_cmd, bug_output) and not run_test(reproduce_cmd, bug_output) and not run_test(reproduce_cmd, bug_output):
        return "bug not reproduced when reducing", ghc_flags
    else:
        while True:
            # Read the original test file lines
            with open(testpath, "r") as f:
                lines = f.readlines()

            # Strip any extra whitespace or newlines
            lines = [line.rstrip("\n") for line in lines]

            # Begin minimizing the test case by removing lines
            minimized_lines, init_step = minimize_testcase(lines, bug_output, testpath, reproduce_cmd)

            # Further minimize by removing multiple lines at once
            further_minimized_lines = further_minimize_testcase(minimized_lines, bug_output, testpath, reproduce_cmd)

            # Restore the minimized test case in the file
            with open(testpath, "w") as f:
                f.write("\n".join(further_minimized_lines))

            n = len(further_minimized_lines)
            step = max(n // 2, 1)
            if step == init_step:
                print("reducing haskell finished")
                break

        reduced_hs = "\n".join(further_minimized_lines)
        reduced_flags = minimize_ghc_flags(testpath, ghc_bin, ghc_flags, bug_output)

        return reduced_hs, reduced_flags


if __name__ == "__main__":

    # Define the path to the test Haskell file; move the reproducer here first
    # (best to also copy any sibling modules to /tmp, though fusion output is
    # always a single self-contained module).
    testpath = "/tmp/test.hs"

    # ghc binary — resolved the same way the driver does
    ghc_bin = _resolve_ghc()

    # Extra flags (beyond the fixed -fno-code -v0) used to reproduce the
    # crash — empty list is fine, most crashes don't depend on a specific
    # flag combination. Fill this in from the "Command:" line of the bug
    # report if it does (e.g. ["-O2"], ["-XStrict"]).
    ghc_flags = []

    # The expected bug output that we are trying to reproduce — defaults to
    # the first pattern in config.yaml's analysis.crash_patterns so it stays
    # in sync with what the driver actually treats as a crash.
    bug_output = _load_crash_patterns()[0]
    bug_output = 'panic!'
    # bug_output = 'GHC internal error'
    # e.g. bug_output = 'Segmentation fault'
    # e.g. bug_output = 'internal error:'
    # e.g. bug_output = '(core dumped)'

    reduced_hs, reduced_flags = reduce_haskell(testpath, ghc_bin, ghc_flags, bug_output)

    reduced_cmd = " ".join([ghc_bin, *reduced_flags, testpath])

    # auto generate bug report
    report_template = "\nThe following code:\n\n```haskell\n{poc}\n```\n\nResulted in this output:\n```\n{stdouterr}\n```\n\nTo reproduce:\n```\n{config}\n```\n\nGHC Version:\n```\n{ghc_version}\n```\n\nOperating System:\n```\n{os}\n```\n\n*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*\n"

    try:
        ghc_version = subprocess.run(
            [ghc_bin, "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        ghc_version = "unknown"

    os_info = "Official `haskell:latest` Docker image (GHC via ghcup)"

    bug_report = report_template.format(
        poc=reduced_hs,
        stdouterr=stdouterr,
        config=reduced_cmd,
        ghc_version=ghc_version,
        os=os_info,
    )

    print('\033[94m' + bug_report + '\033[0m')
