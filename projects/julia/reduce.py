"""
projects/julia/reduce.py — shrink a Julia reproducer and print a report.

Same pipeline as the other adapters' reducers (see projects/r/reduce.py
and projects/ruby/reduce.py): run_test / minimize_testcase /
further_minimize_testcase / reduce_flags, driven by the constants at the
bottom of this file.

Two Julia-specific rules.

**A deletion that breaks the parse is rejected.** Julia parses a script
whole before running any of it, so removing the `end` of a `for` or a
`function` leaves a file that dies with a syntax error — which a reducer
reads as "no longer reproduces" and stops early, keeping lines it did not
need. `Meta.parseall` on the file text answers the question without
running anything.

**The execution axes are part of the configuration.** `--check-bounds`,
the optimisation level, `--compile` and `--inline` decide which parts of
the compiler a program goes through, and several reports appear at only
one setting; they are reduced alongside the file rather than left at
whatever the fuzzer drew. `--check-bounds=no` is where a wrong index
reaches memory, so it is the last thing dropped.
"""

import os
import subprocess

stdouterr = None


def parses(testpath, julia):
    """True when Julia can parse the file. `Meta.parseall` builds the
    whole expression tree and raises on a syntax error without running
    a single line of the program."""
    # `parseall` does not throw on an unterminated block: it returns a
    # `:toplevel` expression with an `:incomplete` (or `:error`) node in
    # it, so both have to be looked for by hand.
    prog = ('ex = try; Meta.parseall(read(ARGS[1], String); filename=ARGS[1]); '
            'catch; exit(1); end; '
            'for a in ex.args; if a isa Expr && (a.head === :incomplete || '
            'a.head === :error); exit(1); end; end; exit(0)')
    try:
        r = subprocess.run(
            [julia, "--startup-file=no", "--color=no", "-e", prog, testpath],
            capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception:
        return False


def run_test(cmd, bug_output, timeout=90):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=timeout)
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


def minimize_testcase(lines, bug_output, testpath, cmd, julia):
    i = len(lines) - 1
    while i >= 0:
        candidate = lines[:i] + lines[i + 1:]
        _write(testpath, candidate)
        if candidate and parses(testpath, julia) and run_test(cmd, bug_output):
            lines = candidate
            print(f"  removed line {i + 1}, {len(lines)} left")
        i -= 1
    _write(testpath, lines)
    return lines


def further_minimize_testcase(lines, bug_output, testpath, cmd, julia):
    size = max(len(lines) // 2, 1)
    while size >= 1:
        i = 0
        while i + size <= len(lines):
            candidate = lines[:i] + lines[i + size:]
            _write(testpath, candidate)
            if candidate and parses(testpath, julia) and run_test(cmd, bug_output):
                lines = candidate
                print(f"  removed {size} lines at {i + 1}, {len(lines)} left")
            else:
                i += 1
        size //= 2
    _write(testpath, lines)
    return lines


def reduce_flags(flags, bug_output, testpath, julia, env):
    """Drop execution axes one at a time. `--check-bounds=no` is tried
    last: with bounds checking on, an out-of-bounds write becomes a
    BoundsError instead of the memory error being reported."""
    order = sorted(flags, key=lambda f: f == "--check-bounds=no")
    kept = list(flags)
    for flag in order:
        trial = [f for f in kept if f != flag]
        if run_test(f"{env}{julia} --startup-file=no --color=no "
                    f"{' '.join(trial)} {testpath}", bug_output):
            kept = trial
            print(f"  dropped flag {flag}")
    return kept


def reduce_julia(testpath, julia, flags, bug_output, env):
    with open(testpath) as f:
        lines = f.read().splitlines()
    cmd = f"{env}{julia} --startup-file=no --color=no {' '.join(flags)} {testpath}"
    if not run_test(cmd, bug_output):
        print("The bug does not reproduce as given; check the julia path, "
              "the flags and bug_output.")
        return lines, flags
    print(f"Reducing {len(lines)} lines ...")
    lines = minimize_testcase(lines, bug_output, testpath, cmd, julia)
    lines = further_minimize_testcase(lines, bug_output, testpath, cmd, julia)
    lines = minimize_testcase(lines, bug_output, testpath, cmd, julia)
    flags = reduce_flags(flags, bug_output, testpath, julia, env)
    return lines, flags


if __name__ == "__main__":
    root = "/home/fuzz/WorkSpace/fusion-fuzz"
    testpath = "/tmp/ffl_repro.jl"
    julia = f"{root}/projects/julia/julia-src/julia"
    env = ("JULIA_DEPOT_PATH=/tmp/ffl-julia-depot JULIA_PKG_OFFLINE=true "
           "JULIA_NUM_THREADS=1 JULIA_LOAD_PATH=@stdlib HOME=/tmp ")

    # From the bundle's test.sh.
    flags = ["--check-bounds=no", "-O2"]

    # A distinctive fragment of the report: the assertion text, the
    # "Internal error" line, or the LLVM ERROR message.
    bug_output = "Internal error: encountered unexpected error in runtime"

    lines, flags = reduce_julia(testpath, julia, flags, bug_output, env)
    poc = "\n".join(lines)
    commit = subprocess.run(f"cd {root}/projects/julia/julia-src && git rev-parse HEAD",
                            shell=True, capture_output=True, text=True).stdout.strip()
    report = f"""
The following program:

```julia
{poc}
```

makes Julia report:

```
{(stdouterr or '')[:3000]}
```

To reproduce:

```
julia --startup-file=no {' '.join(flags)} repro.jl
```

Commit:

```
{commit}
```

Build configuration:

```
make FORCE_ASSERTIONS=1 JULIA_BUILD_MODE=release
```

Operating System:

```
Ubuntu 24.04 host, Docker ffe-julia:latest
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""
    print(report)
    with open("/tmp/ffl_julia_report.md", "w") as f:
        f.write(report)
    print("report written to /tmp/ffl_julia_report.md")
