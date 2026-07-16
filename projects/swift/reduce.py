import subprocess

stdouterr = None


def run_test(cmd, bug_output, timeout=15):
    """Run the reproduce command and check whether bug_output appears in the
    combined stdout/stderr."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            errors="replace", timeout=timeout,
        )
    except Exception:
        return False

    combined = result.stdout + result.stderr
    if bug_output in combined:
        global stdouterr
        if stdouterr is None:
            stdouterr = result.stderr
        return True
    return False


def minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    print("Reducing... this may take a while.")
    n = len(lines)
    step = max(n // 2, 1)
    init_step = step

    while step > 0:
        print(f"Current step: {step}, lines: {n}")
        for i in range(0, n, step):
            temp_lines = lines[:i] + lines[i + step:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
        else:
            step //= 2

    return lines, init_step


def further_minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    n = len(lines)
    for count in range(2, 6):
        for i in range(n - count + 1):
            temp_lines = lines[:i] + lines[i + count:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
    return lines


def reduce_flags(flags, bug_output, testpath, swift_bin, env_prefix):
    """Try removing flags one at a time (e.g. -Onone, -sil-verify-all)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{swift_bin} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_swift(testpath, swift_bin, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{swift_bin} {' '.join(flags)} {testpath}"

    if not (run_test(reproduce_cmd, bug_output) or
            run_test(reproduce_cmd, bug_output) or
            run_test(reproduce_cmd, bug_output)):
        return "bug not reproduced when reducing", flags

    while True:
        with open(testpath, "r") as f:
            lines = f.readlines()
        lines = [line.rstrip('\n') for line in lines]

        minimized_lines, init_step = minimize_testcase(
            lines, bug_output, testpath, reproduce_cmd)
        further_minimized_lines = further_minimize_testcase(
            minimized_lines, bug_output, testpath, reproduce_cmd)

        with open(testpath, "w") as f:
            f.write("\n".join(further_minimized_lines))

        n = len(further_minimized_lines)
        step = max(n // 2, 1)
        if step == init_step:
            print("Reducing Swift source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, swift_bin, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


if __name__ == "__main__":
    # Path to the crashing test case — copy it here (or point directly at a
    # bug's test.swift under output/bugs/swift/<bug_dir>/) before running.
    testpath = "/tmp/test.swift"

    # The full binary invocation, copied from the bug's test.sh (everything
    # before the flags/source file). 'swift' is pre-installed in PATH inside
    # the ffl-swift container. Plain "swift" runs the file via its -interpret
    # mode; use "swift -frontend" instead if the bug's test.sh invokes the
    # frontend directly (in which case `flags` must include a mode flag like
    # -typecheck/-emit-silgen/-emit-sil/-emit-ir/-c).
    swift_bin = "swift"

    # Flags that reproduced the crash — copy these from the bug's test.sh
    # (the tokens after `swift_bin`, before the source file). Order doesn't
    # matter; each is tried for removal independently.
    flags = []

    # Matches projects/swift/driver.py's execution environment, so that
    # ASAN/UBSAN crashes reproduce and are printed rather than silently
    # continuing.
    asan_opts = "abort_on_error=1:detect_leaks=0:symbolize=1:detect_stack_use_after_return=1"
    ubsan_opts = "print_stacktrace=1:halt_on_error=1"
    env_prefix = f"ASAN_OPTIONS='{asan_opts}' UBSAN_OPTIONS='{ubsan_opts}' "

    # The string to look for in swift's output to confirm the bug.
    # Examples:
    #   "Please submit a bug report"
    #   "Assertion failed: ..."
    #   "While evaluating request ..."
    bug_output = "Assertion"

    reduced_src, reduced_flags = reduce_swift(testpath, swift_bin, flags, bug_output, env_prefix)

    if reduced_src == "bug not reproduced when reducing":
        print(f"Error: {reduced_src}. Check that `swift_bin`/`flags` "
              f"reproduce the crash on their own first, and that "
              f"`bug_output` matches the crash text.")
        raise SystemExit(1)

    version_result = subprocess.run(
        f"{swift_bin} --version", shell=True, capture_output=True, text=True)
    swift_version = version_result.stdout.strip()

    reproduce_cmd = f"{env_prefix}{swift_bin} {' '.join(reduced_flags)} {testpath}"

    report_template = """
The following code:

```swift
{poc}
```

Resulted in this output:
```
{stdouterr}
```

To reproduce:
```
{cmd}
```

Compiler version:
```
{version}
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""

    bug_report = report_template.format(
        poc=reduced_src,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=swift_version,
    )

    print('\033[94m' + bug_report + '\033[0m')
