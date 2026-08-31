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


def reduce_flags(flags, bug_output, testpath, lfortran_bin, env_prefix):
    """Try removing lfortran flags one at a time (e.g. --show-asr, -g)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{lfortran_bin} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_lfortran(testpath, lfortran_bin, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{lfortran_bin} {' '.join(flags)} {testpath}"

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
            print("Reducing lfortran source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, lfortran_bin, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


if __name__ == "__main__":
    # Path to the crashing test case — copy it here (or point directly at a
    # bug's test.<ext> under output/bugs/lfortran/<bug_dir>/) before running.
    testpath = "/tmp/test.f90"

    lfortran_bin = "lfortran"

    # Flags that reproduced the crash — copy these from the bug's test.sh
    # (the tokens after "lfortran", before the source file). Order doesn't
    # matter; each is tried for removal independently. See
    # projects/lfortran/driver.py's MODES/STD_VALUES/MISC_FLAGS for the
    # full set of flags the fuzzer itself picks from.
    flags = ["--show-asr", "--std=f23"]
    flags = []

    # Matches projects/lfortran/driver.py's execution environment: caps
    # address space so OOM-y inputs abort cleanly instead of ballooning.
    env_prefix = "ulimit -v 3145728; "

    # The string to look for in lfortran's output to confirm the bug. See
    # projects/lfortran/config.yaml's analysis.crash_patterns for the full
    # list the fuzzer itself matches against. Examples:
    #   "Internal Compiler Error"
    #   "LCOMPILERS_ASSERT failed"
    #   "Segfault: Signal SIGSEGV"
    #   "Abort: Signal SIGABRT"
    bug_output = "Internal Compiler Error"

    reduced_src, reduced_flags = reduce_lfortran(testpath, lfortran_bin, flags, bug_output, env_prefix)

    version_result = subprocess.run(
        f"{lfortran_bin} --version", shell=True, capture_output=True, text=True)
    lfortran_version = version_result.stdout.strip()

    reproduce_cmd = f"{env_prefix}{lfortran_bin} {' '.join(reduced_flags)} ./{os.path.basename(testpath)}"

    src_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "lfortran-src")
    commit_result = subprocess.run(
        f"cd {src_dir} && git rev-parse HEAD",
        shell=True, capture_output=True, text=True)
    commit = (commit_result.stdout or "").strip() or "unknown"

    report_template = """
The following code:

```fortran
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

Commit:
```
{commit}
```

Build configuration:
```
{build_config}
```

Operating System:
```
{os_desc}
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""

    bug_report = report_template.format(
        poc=reduced_src,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=lfortran_version,
        commit=commit,
        build_config="cmake -DCMAKE_BUILD_TYPE=Debug -DWITH_LLVM=yes",
        os_desc="Ubuntu 22.04 Host, Docker fusion-fuzz-lfortran:latest",
    )

    print('\033[94m' + bug_report + '\033[0m')
