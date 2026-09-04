"""
projects/go/reduce.py — shrink a Go ICE reproducer and generate a report.

A fused program is two whole source files stitched together; the ICE usually
needs a handful of their lines. Go maintainers will not act on a 500-line
reproducer, so this narrows it before the bug is filed, reduces the gcflags
set to the ones that are actually required, and prints a report ready to
paste into https://github.com/golang/go/issues.

Structured like the other adapters' reducers (see projects/gcc/reduce.py and
projects/rust/reduce.py): the same run_test / minimize_testcase /
further_minimize_testcase / reduce_flags / reduce_go pipeline, driven by the
constants under `if __name__ == "__main__"` at the bottom of this file.

Three things here differ from the shared shape, all forced by the toolchain.

**`go build` compiles a directory, not a file.** Every other reducer rewrites
one source path and re-runs the compiler on it. Go needs a module around the
file -- without a go.mod the build fails with "cannot find main module"
before it ever looks at the program, which a reducer would read as "no longer
reproduces" and happily delete the whole file. So the testcase is written as
main.go into a scratch module directory that this file creates and maintains.

**GOOS/GOARCH belong in the env prefix, never in `flags`.** They are not
optional decoration: several of the bugs this reducer was written against
fire only on riscv64. Putting them where reduce_flags can see them would let
it strip the target and conclude the crash had stopped reproducing.

**Unused imports are a compile error**, so removing an import and removing its
last user are each individually invalid and line-at-a-time deletion keeps
both forever. drop_imports_with_users removes them as one edit.

As in projects/gcc/reduce.py, `bug_output` should be the **specific ICE
message**, not a bare "internal compiler error": a reduction that trades the
original ICE for a different one has found a different bug, and only the
specific message rejects it. This is not hypothetical here -- dropping
`//go:uintptrescapes` from one reproducer turned a ZeroLoop ICE into a
MoveLoop ICE, a genuinely different defect.

Usage: edit the constants at the bottom, then
    python3 projects/go/reduce.py
"""

import os
import re
import shutil
import subprocess
import tempfile

GO_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "go-src")
GO_BIN = os.path.join(GO_SRC, "bin", "go")

_SIG_RE = re.compile(r"internal compiler error:\s*([^\n]+)")

stdouterr = None


def run_test(cmd, bug_output, timeout=120):
    """Run the reproduce command and check whether bug_output appears in the
    combined stdout/stderr.

    A timeout counts as "did not reproduce". Some reductions send the type
    checker into unbounded instantiation, and keeping such a line because the
    run never finished would stall every later round.
    """
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
            stdouterr = combined
        return True
    return False


def signature_of(output):
    """The ICE message, which is what `bug_output` should be set to."""
    m = _SIG_RE.search(output or "")
    return m.group(1).strip() if m else None


def make_module(testpath, go_directive="1.21"):
    """Put *testpath* in a scratch module and return (moduledir, mainfile).

    `go build .` needs a go.mod; a bug bundle's directory has none, and it
    also holds parent_a/parent_b/min alongside test, which would all be
    compiled into the same package. Copying the one file out into a fresh
    module is what makes the reduction measure the program we care about.
    """
    moduledir = tempfile.mkdtemp(prefix="ffl_go_reduce_")
    with open(os.path.join(moduledir, "go.mod"), "w") as f:
        f.write(f"module fflreduce\n\ngo {go_directive}\n")
    mainfile = os.path.join(moduledir, "main.go")
    shutil.copyfile(testpath, mainfile)
    return moduledir, mainfile


def build_cmd(moduledir, go_bin, flags, env_prefix):
    """The command that compiles the scratch module."""
    gcflags = f" -gcflags='{' '.join(flags)}'" if flags else ""
    return f"cd {moduledir} && {env_prefix}{go_bin} build -o /dev/null{gcflags} ."


def minimize_testcase(lines, bug_output, mainfile, reproduce_cmd):
    """Remove lines in shrinking chunks while the ICE still reproduces."""
    print("Reducing... this may take a while.")
    n = len(lines)
    step = max(n // 2, 1)
    init_step = step

    while step > 0:
        print(f"Current step: {step}")
        for i in range(0, n, step):
            temp_lines = lines[:i] + lines[i + step:]
            with open(mainfile, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
        else:
            step //= 2

    return lines, init_step


def further_minimize_testcase(lines, bug_output, mainfile, reproduce_cmd):
    """Second pass: drop 2-5 adjacent lines at a time.

    The chunked pass above cannot remove a construct whose lines are not a
    power-of-two-aligned run, which is most of them.
    """
    n = len(lines)
    for count in range(2, 6):
        for i in range(n - count + 1):
            temp_lines = lines[:i] + lines[i + count:]
            with open(mainfile, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
    return lines


def strip_comments(lines, bug_output, mainfile, reproduce_cmd):
    """Drop comments once the line reduction has settled.

    Line deletion cannot touch a trailing comment, so a reduced fused program
    arrives still carrying fusion-fuzz's own markers (`// declaration fusion`,
    `// state fusion`) and the seed's leftover testsuite directives
    (`// GC_ERROR "..."`). Those are internals; a report that shows them to a
    Go maintainer is asking them to read our plumbing. Each removal is still
    checked against the oracle, so a comment that somehow matters is kept --
    `//go:` directives are load-bearing and survive for exactly that reason.
    """
    out = list(lines)
    for i, line in enumerate(out):
        if "//" not in line:
            continue
        stripped = line[:line.index("//")].rstrip()
        if not stripped:
            continue                      # whole-line comment: handled below
        trial = out[:i] + [stripped] + out[i + 1:]
        with open(mainfile, "w") as f:
            f.write("\n".join(trial))
        if run_test(reproduce_cmd, bug_output):
            out = trial

    trial = [l for l in out if l.strip() and not l.strip().startswith("//")]
    with open(mainfile, "w") as f:
        f.write("\n".join(trial))
    if run_test(reproduce_cmd, bug_output):
        out = trial

    with open(mainfile, "w") as f:
        f.write("\n".join(out))
    return out


def drop_imports_with_users(lines, bug_output, mainfile, reproduce_cmd):
    """Remove an import together with every line that mentions it.

    Go makes an unused import a compile *error*, which defeats line-at-a-time
    deletion: dropping the import alone fails to build, and dropping its last
    user alone leaves the import unused and also fails to build. Neither step
    is individually valid, so the chunk passes keep both forever -- a reduced
    testcase arrives still carrying `import "reflect"` and whatever function
    happened to call it.

    Removing the import and its users as one edit gets past that. The oracle
    still gates it, so an import the crash actually needs survives.
    """
    out = list(lines)
    for _ in range(8):                       # a dropped user can free another
        imports = []
        for i, l in enumerate(out):
            m = (re.match(r'\s*(?:[\w.]+\s+)?"([^"]+)"\s*$', l)
                 or re.match(r'\s*import\s+(?:[\w.]+\s+)?"([^"]+)"', l))
            if m:
                imports.append((i, m.group(1)))
        if not imports:
            break
        for idx, path in imports:
            pkg = path.rsplit("/", 1)[-1]
            users = {j for j, l in enumerate(out)
                     if j != idx and re.search(rf'\b{re.escape(pkg)}\.', l)}
            trial = [l for j, l in enumerate(out) if j != idx and j not in users]
            if not trial:
                continue
            with open(mainfile, "w") as f:
                f.write("\n".join(trial))
            if run_test(reproduce_cmd, bug_output):
                out = trial
                break
        else:
            break

    # An emptied `import ( )` block is itself removable.
    trial, skip = [], False
    for l in out:
        if re.match(r"\s*import\s*\($", l):
            skip = True
            continue
        if skip and re.match(r"\s*\)\s*$", l):
            skip = False
            continue
        if not skip:
            trial.append(l)
    if trial != out:
        with open(mainfile, "w") as f:
            f.write("\n".join(trial))
        if run_test(reproduce_cmd, bug_output):
            out = trial

    with open(mainfile, "w") as f:
        f.write("\n".join(out))
    return out


def reduce_flags(flags, bug_output, moduledir, go_bin, env_prefix):
    """Try removing gcflags one at a time (e.g. -N, -d=ssa/check/on).

    Worth doing even when it strips everything: a Go ICE that survives with
    no gcflags at all is one a user hits with a plain `go build`, which is a
    materially stronger report than one that needs -d=panic to become
    visible. base.FatalfAt suppresses the ICE message when ordinary errors
    were already reported, so -d=panic can manufacture output a real user
    would never see, and -d=ssa/check/on can surface internal-consistency
    failures that never reach an ordinary build.
    """
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = build_cmd(moduledir, go_bin, trial, env_prefix)
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_go(testpath, go_bin, flags, bug_output, env_prefix=""):
    moduledir, mainfile = make_module(testpath)
    try:
        reproduce_cmd = build_cmd(moduledir, go_bin, flags, env_prefix)

        if not (run_test(reproduce_cmd, bug_output) or
                run_test(reproduce_cmd, bug_output) or
                run_test(reproduce_cmd, bug_output)):
            return "bug not reproduced when reducing", flags

        while True:
            with open(mainfile, "r", errors="replace") as f:
                lines = f.read().splitlines()

            minimized_lines, init_step = minimize_testcase(
                lines, bug_output, mainfile, reproduce_cmd)
            further_minimized_lines = further_minimize_testcase(
                minimized_lines, bug_output, mainfile, reproduce_cmd)
            further_minimized_lines = strip_comments(
                further_minimized_lines, bug_output, mainfile, reproduce_cmd)
            further_minimized_lines = drop_imports_with_users(
                further_minimized_lines, bug_output, mainfile, reproduce_cmd)

            with open(mainfile, "w") as f:
                f.write("\n".join(further_minimized_lines))

            n = len(further_minimized_lines)
            step = max(n // 2, 1)
            if step == init_step:
                print("Reducing Go source finished.")
                break

        reduced_src = "\n".join(further_minimized_lines)

        print("Reducing flags...")
        reduced_flags = reduce_flags(flags, bug_output, moduledir, go_bin, env_prefix)
        print(f"Reduced flags: {reduced_flags}")

        # The reduced program has to be re-checked against the reduced flag
        # set: the two were minimized independently, and stdouterr must come
        # from the combination the report actually tells people to run.
        global stdouterr
        stdouterr = None
        with open(mainfile, "w") as f:
            f.write(reduced_src)
        run_test(build_cmd(moduledir, go_bin, reduced_flags, env_prefix), bug_output)

        return reduced_src, reduced_flags
    finally:
        shutil.rmtree(moduledir, ignore_errors=True)


if __name__ == "__main__":
    # Path to the crashing test case — copy it to /tmp first (or point
    # directly at a bug's test.go under output/bugs/go/<bug_dir>/). It is
    # copied into a scratch module; the original is not modified.
    testpath = "/tmp/test.go"

    go_bin = GO_BIN

    # The gcflags that reproduced the crash — copy these from the bug's
    # test.sh (the tokens inside -gcflags='...'). Order doesn't matter; each
    # is tried for removal independently.
    flags = ["-d=panic", "-d=ssa/check/on"]

    # Matches projects/go/driver.py's execution environment. GOOS/GOARCH go
    # here rather than in `flags` on purpose — see the module docstring; a
    # cross-target ICE stops reproducing the moment the target is dropped.
    # GOCACHE is shared so the target's standard library is not rebuilt on
    # every one of the hundreds of runs this makes.
    env_prefix = (
        "ulimit -v 4194304; ulimit -c 0; "
        f"GOROOT={GO_SRC} "
        f"GOCACHE={os.path.join(os.path.dirname(os.path.dirname(GO_SRC)), '..', '.fused', 'go-cache')} "
        "GOOS=linux GOARCH=amd64 GOPROXY=off GOFLAGS=-mod=mod CGO_ENABLED=0 "
    )

    # The string to look for in Go's output to confirm the bug.
    #
    # Use the *specific* ICE message, not a generic marker: a reduction that
    # trades this ICE for a different one has found a different bug, and only
    # the specific message rejects it. Get it from the bug's test.out, or
    # from signature_of() on a run of the unreduced input.
    #   "MoveLoop too small"
    #   "hasUncommon: methods not computed"
    bug_output = "hasUncommon: methods not computed"

    reduced_src, reduced_flags = reduce_go(
        testpath, go_bin, flags, bug_output, env_prefix)

    version_result = subprocess.run(
        f"GOROOT={GO_SRC} {go_bin} version", shell=True,
        capture_output=True, text=True)
    go_version = (version_result.stdout or "").strip()

    commit_result = subprocess.run(
        f"cd {GO_SRC} && git rev-parse HEAD", shell=True,
        capture_output=True, text=True)
    commit = (commit_result.stdout or "").strip() or "unknown"

    # The target matters to the report, so lift it back out of env_prefix.
    m = re.search(r"GOOS=(\S+)\s+GOARCH=(\S+)", env_prefix)
    target = f"GOOS={m.group(1)} GOARCH={m.group(2)} " if m else ""
    gcflags = f" -gcflags='{' '.join(reduced_flags)}'" if reduced_flags else ""
    reproduce_cmd = f"{target}go build{gcflags} ."

    report_template = """
The following code:

```go
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
        version=go_version,
        commit=commit,
        build_config="cd src && ./make.bash (GOROOT_BOOTSTRAP=go1.26.0)",
        os_desc="Ubuntu 24.04 Host, Docker fusion-fuzz-go:latest",
    )

    print('\033[94m' + bug_report + '\033[0m')
