"""
projects/gcc/reduce.py — shrink a GCC ICE reproducer and generate a report.

A fused program is two whole source files stitched together; the ICE usually
needs a handful of their lines. GCC maintainers will not act on a 460-line
reproducer, so this narrows it before the bug is filed, reduces the flag set
to the ones that are actually required, and prints a report ready to paste
into https://gcc.gnu.org/bugzilla/.

Structured like the other adapters' reducers (see projects/clang/reduce.py
and projects/flang/reduce.py): the same run_test / minimize_testcase /
further_minimize_testcase / reduce_flags / reduce_gcc pipeline, driven by the
constants under `if __name__ == "__main__"` at the bottom of this file.

One thing here is stricter than the shared shape, deliberately. `bug_output`
elsewhere is any distinctive substring; for GCC it should be the **ICE
message itself**, because a reduction that trades the original ICE for a
different one has found a different bug. Matching on the specific message
rejects that; matching on a bare "internal compiler error" would accept it
and the report would describe code that no longer triggers what was
observed.

Usage: edit the constants at the bottom, then
    python3 projects/gcc/reduce.py
"""

import os
import re
import subprocess

GCC_INSTALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcc-install")
GCC_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcc-src")

_SIG_RE = re.compile(r"internal compiler error:\s*([^\n]+)")

stdouterr = None


def run_test(cmd, bug_output, timeout=120):
    """Run the reproduce command and check whether bug_output appears in the
    combined stdout/stderr.

    The timeout is far longer than the other reducers use: GCC with
    `--enable-checking=yes,extra,rtl` is slow, and a reduction step that
    times out would be recorded as "no longer reproduces" and the line
    wrongly kept.
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


def minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    """Remove lines in shrinking chunks while the ICE still reproduces."""
    print("Reducing... this may take a while.")
    n = len(lines)
    step = max(n // 2, 1)
    init_step = step

    while step > 0:
        print(f"Current step: {step}")
        for i in range(0, n, step):
            temp_lines = lines[:i] + lines[i + step:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
        else:
            step //= 2

    return lines, init_step


def further_minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    """Second pass: drop 2-5 adjacent lines at a time.

    The chunked pass above cannot remove a construct whose lines are not a
    power-of-two-aligned run, which is most of them.
    """
    n = len(lines)
    for count in range(2, 6):
        for i in range(n - count + 1):
            temp_lines = lines[:i] + lines[i + count:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
    return lines


def reduce_flags(flags, bug_output, testpath, gcc_bin, env_prefix):
    """Try removing flags one at a time (e.g. -O2, -fno-strict-aliasing)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{gcc_bin} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_gcc(testpath, gcc_bin, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{gcc_bin} {' '.join(flags)} {testpath}"

    if not (run_test(reproduce_cmd, bug_output) or
            run_test(reproduce_cmd, bug_output) or
            run_test(reproduce_cmd, bug_output)):
        return "bug not reproduced when reducing", flags

    while True:
        with open(testpath, "r", errors="replace") as f:
            lines = f.read().splitlines()

        minimized_lines, init_step = minimize_testcase(
            lines, bug_output, testpath, reproduce_cmd)
        further_minimized_lines = further_minimize_testcase(
            minimized_lines, bug_output, testpath, reproduce_cmd)

        with open(testpath, "w") as f:
            f.write("\n".join(further_minimized_lines))

        n = len(further_minimized_lines)
        step = max(n // 2, 1)
        if step == init_step:
            print("Reducing GCC source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, gcc_bin, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


def _gcc_bin_for(testpath):
    """g++ for C++ sources, gcc otherwise."""
    binary = "g++" if testpath.endswith((".cc", ".cpp", ".C", ".cxx", ".c++")) else "gcc"
    return os.path.join(GCC_INSTALL, "bin", binary)


if __name__ == "__main__":
    # Path to the crashing test case — copy it to /tmp first (or point
    # directly at a bug's test.<ext> under output/bugs/gcc/<bug_dir>/).
    # It is rewritten in place as the reduction proceeds.
    testpath = "/tmp/test.c"

    gcc_bin = _gcc_bin_for(testpath)

    # Flags that reproduced the crash — copy these from the bug's test.sh
    # (the tokens after "gcc"/"g++", before the source file). Order doesn't
    # matter; each is tried for removal independently.
    flags = ["-c", "-o", "/dev/null", "-O2"]

    # Matches projects/gcc/driver.py's execution environment: caps address
    # space so a runaway input aborts cleanly instead of taking the machine
    # with it, and suppresses core dumps.
    env_prefix = "ulimit -v 4194304; ulimit -c 0; "

    # The string to look for in GCC's output to confirm the bug.
    #
    # Use the *specific* ICE message, not a generic marker: a reduction that
    # trades this ICE for a different one has found a different bug, and only
    # the specific message rejects it. Get it from the bug's test.out, or
    # from signature_of() on a run of the unreduced input.
    #   "in verify_gimple_in_cfg, at tree-cfg.cc:5570"
    #   "Segmentation fault"
    bug_output = "internal compiler error:"

    reduced_src, reduced_flags = reduce_gcc(
        testpath, gcc_bin, flags, bug_output, env_prefix)

    version_result = subprocess.run(
        f"{gcc_bin} --version", shell=True, capture_output=True, text=True)
    gcc_version = (version_result.stdout or "").strip()

    commit_result = subprocess.run(
        f"cd {GCC_SRC} && git rev-parse HEAD", shell=True,
        capture_output=True, text=True)
    commit = (commit_result.stdout or "").strip() or "unknown"

    build_config = ("./configure --enable-languages=c,c++ --disable-bootstrap "
                    "--enable-checking=yes,extra,rtl --disable-multilib "
                    "--disable-libsanitizer --disable-nls --disable-werror")

    reproduce_cmd = (f"{gcc_bin} {' '.join(reduced_flags)} "
                     f"./{os.path.basename(testpath)}")

    lang = "cpp" if testpath.endswith((".cc", ".cpp", ".C", ".cxx", ".c++")) else "c"

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
        lang=lang,
        poc=reduced_src,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=gcc_version,
        commit=commit,
        build_config=build_config,
        os_desc="Ubuntu 22.04 Host, Docker fusion-fuzz-gcc:latest",
    )

    print('\033[94m' + bug_report + '\033[0m')
