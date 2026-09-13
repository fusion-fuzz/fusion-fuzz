"""
projects/r/reduce.py — shrink an R reproducer and print a report.

Same pipeline as the other adapters' reducers (see
projects/ruby/reduce.py and projects/cpython/reduce.py): run_test /
minimize_testcase / further_minimize_testcase / reduce_flags, driven by
the constants at the bottom of this file.

Two R-specific rules.

**A deletion that breaks the parse is rejected**, checked with
`parse(file=)` before the program is run. R reads a script top to bottom
but parses it whole, so removing the `}` of a block leaves a file that
fails at parse time — which a reducer would read as "no longer
reproduces" and stop early.

**The JIT level is part of the configuration.** `R_ENABLE_JIT` (0-3)
decides how much of the program the byte-code compiler compiles, and
several of these reports only appear at one level, so it is reduced
alongside the flags rather than left at whatever the fuzzer drew.
"""

import os
import subprocess

stdouterr = None


def parses(testpath, rscript):
    try:
        r = subprocess.run(
            f'{rscript} --vanilla -e \'invisible(parse(file="{testpath}"))\'',
            shell=True, capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def run_test(cmd, bug_output, timeout=40):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                encoding="iso-8859-1", timeout=timeout)
    except Exception:
        return False
    hit = bug_output in (result.stdout or "") or bug_output in (result.stderr or "")
    if hit:
        global stdouterr
        if stdouterr is None:
            stdouterr = (result.stderr or "") + (result.stdout or "")
    return hit


def _write(testpath, lines):
    with open(testpath, "w") as f:
        f.write("\n".join(lines) + "\n")


def minimize_testcase(lines, bug_output, testpath, cmd, rscript):
    i = len(lines) - 1
    while i >= 0:
        candidate = lines[:i] + lines[i + 1:]
        _write(testpath, candidate)
        if candidate and parses(testpath, rscript) and run_test(cmd, bug_output):
            lines = candidate
            print(f"  removed line {i + 1}, {len(lines)} left")
        i -= 1
    _write(testpath, lines)
    return lines


def further_minimize_testcase(lines, bug_output, testpath, cmd, rscript):
    size = max(len(lines) // 2, 1)
    while size >= 1:
        i = 0
        while i + size <= len(lines):
            candidate = lines[:i] + lines[i + size:]
            _write(testpath, candidate)
            if candidate and parses(testpath, rscript) and run_test(cmd, bug_output):
                lines = candidate
                print(f"  removed {size} lines at {i + 1}, {len(lines)} left")
            else:
                i += 1
        size //= 2
    _write(testpath, lines)
    return lines


def reduce_config(jit, flags, bug_output, testpath, rscript):
    """Drop flags one at a time, then find the lowest JIT level that
    still reports the bug (0 is the plain AST interpreter, so a report
    that survives at 0 has nothing to do with the byte-code compiler)."""
    kept = list(flags)
    for flag in list(kept):
        trial = [f for f in kept if f != flag]
        if run_test(f"R_ENABLE_JIT={jit} {rscript} --vanilla {' '.join(trial)} {testpath}",
                    bug_output):
            kept = trial
            print(f"  dropped flag {flag}")
    for level in ("0", "1", "2", "3"):
        if run_test(f"R_ENABLE_JIT={level} {rscript} --vanilla {' '.join(kept)} {testpath}",
                    bug_output):
            print(f"  reproduces at R_ENABLE_JIT={level}")
            return level, kept
    return jit, kept


def reduce_r(testpath, rscript, jit, flags, bug_output):
    with open(testpath) as f:
        lines = f.read().splitlines()
    cmd = f"R_ENABLE_JIT={jit} {rscript} --vanilla {' '.join(flags)} {testpath}"
    if not run_test(cmd, bug_output):
        print("The bug does not reproduce as given; check the Rscript path, "
              "R_ENABLE_JIT, flags and bug_output.")
        return lines, jit, flags
    print(f"Reducing {len(lines)} lines ...")
    lines = minimize_testcase(lines, bug_output, testpath, cmd, rscript)
    lines = further_minimize_testcase(lines, bug_output, testpath, cmd, rscript)
    lines = minimize_testcase(lines, bug_output, testpath, cmd, rscript)
    jit, flags = reduce_config(jit, flags, bug_output, testpath, rscript)
    return lines, jit, flags


if __name__ == "__main__":
    root = "/home/fuzz/WorkSpace/fusion-fuzz"
    testpath = "/tmp/ffl_repro.R"
    rscript = f"{root}/projects/r/install/bin/Rscript"

    # From the bundle's test.sh.
    jit = "3"
    flags = []

    # A distinctive fragment of the report: the sanitizer summary line,
    # "caught segfault", or the internal-error text.
    bug_output = "caught segfault"

    lines, jit, flags = reduce_r(testpath, rscript, jit, flags, bug_output)
    poc = "\n".join(lines)
    commit = subprocess.run(f"cd {root}/projects/r/r-src && git rev-parse HEAD",
                            shell=True, capture_output=True, text=True).stdout.strip()
    report = f"""
The following program:

```r
{poc}
```

makes R report:

```
{(stdouterr or '')[:3000]}
```

To reproduce:

```
R_ENABLE_JIT={jit} Rscript --vanilla {' '.join(flags)} repro.R
```

Commit:

```
{commit}
```

Build configuration:

```
../configure CC="clang-18 -fsanitize=address -fno-omit-frame-pointer" \\
  --enable-strict-barrier --with-recommended-packages=no --with-x=no
```

Operating System:

```
Ubuntu 24.04 host, Docker ffe-r:latest
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""
    print(report)
    with open("/tmp/ffl_r_report.md", "w") as f:
        f.write(report)
    print("report written to /tmp/ffl_r_report.md")
