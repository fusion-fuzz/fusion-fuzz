import os
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


def reduce_flags(flags, bug_output, testpath, clang_bin, env_prefix):
    """Try removing clang flags one at a time (e.g. -O2, -std=c++17, -Wall)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{clang_bin} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_clang(testpath, clang_bin, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{clang_bin} {' '.join(flags)} {testpath}"

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
            print("Reducing clang source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, clang_bin, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


_CXX_EXTS = (".cpp", ".cc", ".cxx", ".mm")


def _clang_bin_for(testpath):
    ext = os.path.splitext(testpath)[1].lower()
    return "clang++" if ext in _CXX_EXTS else "clang"


def _lang_tag_for(testpath):
    ext = os.path.splitext(testpath)[1].lower()
    if ext in (".cpp", ".cc", ".cxx"):
        return "cpp"
    if ext == ".mm":
        return "objective-c++"
    if ext == ".m":
        return "objective-c"
    return "c"


if __name__ == "__main__":
    # Path to the crashing test case — copy it here (or point directly at a
    # bug's test.<ext> under output/bugs/clang/<bug_dir>/) before running.
    testpath = "/tmp/test.cpp"

    clang_bin = _clang_bin_for(testpath)
    print(clang_bin)

    # Flags that reproduced the crash — copy these from the bug's test.sh
    # (the tokens after "clang"/"clang++", before the source file). Order
    # doesn't matter; each is tried for removal independently.
    flags = ["-S", "-o", "/dev/null", "-O2", "-std=c++17"]
    flags = []

    # Matches projects/clang/driver.py's execution environment: caps address
    # space so OOM-y inputs abort cleanly ("LLVM ERROR: out of memory")
    # instead of ballooning, and sets sanitizer options so ASAN/UBSAN
    # crashes are printed rather than silently continuing.
    env_prefix = ""
        #"ulimit -v 3145728; "
        #"ASAN_OPTIONS='abort_on_error=1:detect_leaks=0:symbolize=1' "
        #"UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' "
    #)

    # The string to look for in clang's output to confirm the bug.
    # Examples:
    #   "Assertion `New->getType() == getType()' failed"
    #   "LLVM ERROR: out of memory"
    #   "Stack dump:"
    #   "Segmentation fault"
    bug_output = "Stack dump:"

    reduced_src, reduced_flags = reduce_clang(testpath, clang_bin, flags, bug_output, env_prefix)

    version_result = subprocess.run(
        f"{clang_bin} --version", shell=True, capture_output=True, text=True)
    clang_version = version_result.stdout.strip()

    reproduce_cmd = f"{env_prefix}{clang_bin} {' '.join(reduced_flags)} ./{os.path.basename(testpath)}"

    lang_tag = _lang_tag_for(testpath)

    report_template = """
The following code:

```{lang}
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
        lang=lang_tag,
        poc=reduced_src,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=clang_version,
    )

    print('\033[94m' + bug_report + '\033[0m')
