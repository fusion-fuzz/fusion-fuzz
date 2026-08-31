import os
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


def reduce_flags(flags, bug_output, testpath, tint_bin, env_prefix):
    """Try removing tint flags one at a time (e.g. -O2, -ffree-form)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{tint_bin} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_tint(testpath, tint_bin, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{tint_bin} {' '.join(flags)} {testpath}"

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
            print("Reducing tint source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, tint_bin, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


if __name__ == "__main__":
    # Path to the crashing shader — copy it to /tmp first (or point directly
    # at a bug's test.wgsl under output/bugs/tint/<bug_dir>/). It is
    # rewritten in place as the reduction proceeds.
    testpath = "/tmp/test.wgsl"

    tint_bin = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "dawn-src", "dawn", "out", "fuzz", "tint")

    # Flags that reproduced the crash — copy these from the bug's test.sh
    # (the tokens after the tint binary, minus the shader path). Order
    # doesn't matter; each is tried for removal independently.
    #
    # --format is the one to keep an eye on: a translation bug lives in one
    # code writer, so removing it changes which backend runs and the crash
    # will simply stop reproducing rather than reduce.
    flags = ["--format", "spirv", "--validate"]

    # Matches projects/tint/driver.py's execution environment.
    env_prefix = ("ASAN_OPTIONS='allocator_may_return_null=1:detect_leaks=0:"
                  "symbolize=1:handle_sigill=1' "
                  "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=0' ")

    # The string to look for in tint's output to confirm the bug.
    #
    # Use the *specific* internal error, not a generic marker: a reduction
    # that trades this ICE for a different one has found a different bug.
    # Take it from the bug's test.out.
    #   "internal compiler error: TINT_ASSERT(expr)"
    #   "TINT_UNREACHABLE"
    #   "AddressSanitizer"
    bug_output = "internal compiler error"

    reduced_src, reduced_flags = reduce_tint(
        testpath, tint_bin, flags, bug_output, env_prefix)

    # tint has no --version flag (only --help); the Dawn commit below is
    # what identifies the build, so the report carries that instead.
    tint_version = "tint (Dawn) — see Commit below"

    src_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "dawn-src", "dawn")
    commit_result = subprocess.run(
        f"cd {src_dir} && git rev-parse HEAD",
        shell=True, capture_output=True, text=True)
    commit = (commit_result.stdout or "").strip() or "unknown"

    build_config = ("cmake -GNinja -DCMAKE_BUILD_TYPE=Debug "
                    "-DDAWN_FETCH_DEPENDENCIES=ON -DDAWN_ALWAYS_ASSERT=ON "
                    "-DTINT_ENABLE_IR_VALIDATION_ASSERTS=ON "
                    "-DDAWN_ENABLE_ASAN=ON -DDAWN_ENABLE_UBSAN=ON")

    reproduce_cmd = (f"tint ./{os.path.basename(testpath)} "
                     f"{' '.join(reduced_flags)}")

    report_template = """
The following shader:

```wgsl
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
        version=tint_version,
        commit=commit,
        build_config=build_config,
        os_desc="Ubuntu 24.04 Host, Docker fusion-fuzz-tint:latest",
    )

    print('\033[94m' + bug_report + '\033[0m')
