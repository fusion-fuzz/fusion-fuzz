"""
projects/typescript/reduce.py — shrink a tsc reproducer and print a report.

Same pipeline as the other adapters' reducers (see
projects/ruby/reduce.py): run_test / minimize_testcase /
further_minimize_testcase / reduce_flags, driven by the constants at the
bottom of this file.

Two things differ from the executing-language reducers.

**No parse gate.** The signal here is a Go panic from the compiler, and
a syntactically broken file simply does not panic, so a deletion that
breaks the parse fails the reproduce test on its own. (A panic *in* the
parser is the exception, and there the broken file is the point.)

**The `// @option:` directives are part of the program.** They are
comments, so line deletion will happily remove them, and that is
correct: a directive the crash does not need should go. But the ones it
does need must survive, which is why they are reduced by the same
line loop rather than by reduce_flags.
"""

import os
import subprocess

stdouterr = None


def run_test(cmd, bug_output, timeout=60):
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


def minimize_testcase(lines, bug_output, testpath, cmd):
    i = len(lines) - 1
    while i >= 0:
        candidate = lines[:i] + lines[i + 1:]
        _write(testpath, candidate)
        if candidate and run_test(cmd, bug_output):
            lines = candidate
            print(f"  removed line {i + 1}, {len(lines)} left")
        i -= 1
    _write(testpath, lines)
    return lines


def further_minimize_testcase(lines, bug_output, testpath, cmd):
    size = max(len(lines) // 2, 1)
    while size >= 1:
        i = 0
        while i + size <= len(lines):
            candidate = lines[:i] + lines[i + size:]
            _write(testpath, candidate)
            if candidate and run_test(cmd, bug_output):
                lines = candidate
                print(f"  removed {size} lines at {i + 1}, {len(lines)} left")
            else:
                i += 1
        size //= 2
    _write(testpath, lines)
    return lines


def reduce_flags(flags, bug_output, testpath, tsc):
    kept = list(flags)
    for flag in list(kept):
        trial = [f for f in kept if f != flag]
        if run_test(f"{tsc} {' '.join(trial)} {testpath}", bug_output):
            kept = trial
            print(f"  dropped flag {flag}")
    return kept


def reduce_ts(testpath, tsc, flags, bug_output):
    with open(testpath) as f:
        lines = f.read().splitlines()
    cmd = f"{tsc} {' '.join(flags)} {testpath}"
    if not run_test(cmd, bug_output):
        print("The bug does not reproduce as given; check the tsc path, flags "
              "and bug_output.")
        return lines, flags
    print(f"Reducing {len(lines)} lines ...")
    lines = minimize_testcase(lines, bug_output, testpath, cmd)
    lines = further_minimize_testcase(lines, bug_output, testpath, cmd)
    lines = minimize_testcase(lines, bug_output, testpath, cmd)
    kept = reduce_flags(flags, bug_output, testpath, tsc)
    return lines, kept


if __name__ == "__main__":
    root = "/home/fuzz/WorkSpace/fusion-fuzz"
    testpath = "/tmp/ffl_repro.ts"
    tsc = f"{root}/projects/typescript/tsc-bin"

    # From the bundle's test.sh (the `--outDir` there is a temp path; drop it).
    flags = ["--noEmit"]

    # A distinctive fragment: the panic message, "Debug failure", or the
    # first compiler frame.
    bug_output = "panic:"

    lines, kept = reduce_ts(testpath, tsc, flags, bug_output)
    poc = "\n".join(lines)
    commit = subprocess.run(
        f"cd {root}/projects/typescript/typescript-src && git rev-parse HEAD",
        shell=True, capture_output=True, text=True).stdout.strip()
    report = f"""
The following program:

```ts
{poc}
```

makes tsc report:

```
{(stdouterr or '')[:3000]}
```

To reproduce:

```
tsc {' '.join(kept)} repro.ts
```

Commit:

```
{commit}
```

Build:

```
cd tsc && go build -o tsc-bin ./cmd/tsc   # go1.26
```

Operating System:

```
Ubuntu 24.04 host, Docker ffe-go:latest
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""
    print(report)
    with open("/tmp/ffl_ts_report.md", "w") as f:
        f.write(report)
    print("report written to /tmp/ffl_ts_report.md")
