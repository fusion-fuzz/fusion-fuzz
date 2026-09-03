"""
projects/triton/reduce.py — shrink a triton-opt crash reproducer and
generate a report.

A fused module is two whole MLIR files stitched together; the failure
usually needs a handful of their operations. Triton maintainers will not
act on a 400-line module, so this narrows it before the bug is filed,
reduces the pass pipeline to the passes actually required, and prints a
report ready to paste into https://github.com/triton-lang/triton/issues.

Structured like the other adapters' reducers (projects/clang/reduce.py,
projects/flang/reduce.py): the same run_test / minimize_testcase /
further_minimize_testcase / reduce_flags / reduce_triton pipeline, driven
by the constants under `if __name__ == "__main__"` at the bottom.

Two things are specific to MLIR and worth knowing before editing those
constants:

  * The pass pipeline matters as much as the module. Reducing
    `-tritongpu-pipeline -canonicalize` to just `-canonicalize` while the
    crash still reproduces is a *better* report — it says the bug is in
    canonicalisation, not in the Triton pass. reduce_flags does that.

  * `bug_output` should be the assertion expression or the UNREACHABLE
    location, not a generic marker. MLIR prints `error:` for every
    ill-typed module, and delta debugging will happily "preserve" a crash
    by producing a module that merely fails to verify.
"""

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


def reduce_flags(flags, bug_output, testpath, triton_opt, env_prefix):
    """Try removing triton-opt passes one at a time (e.g. -O2, -ffree-form)."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{triton_opt} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_triton(testpath, triton_opt, flags, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{triton_opt} {' '.join(flags)} {testpath}"

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
            print("Reducing flang source finished.")
            break

    reduced_src = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, triton_opt, env_prefix)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_src, reduced_flags


if __name__ == "__main__":
    # Path to the crashing module — copy it to /tmp first (or point
    # directly at a bug's test.mlir under output/bugs/triton/<bug_dir>/).
    # It is rewritten in place as the reduction proceeds.
    testpath = "/tmp/test.mlir"

    triton_opt = ("/home/fuzz/WorkSpace/fusion-fuzz/projects/triton/"
                  "triton-build/bin/triton-opt")

    # The pass pipeline that reproduced the crash — copy it from the bug's
    # test.sh (the tokens after "triton-opt" and the input file). Each is
    # tried for removal independently, and a shorter pipeline is a better
    # report: it says which pass is at fault.
    #
    # -allow-unregistered-dialect is kept in the list rather than pinned,
    # so the reduction can tell you whether the bug needs it.
    flags = ["-allow-unregistered-dialect", "-tritongpu-coalesce"]

    # Matches projects/triton/driver.py's execution environment.
    env_prefix = "ulimit -v 4194304; ulimit -c 0; "

    # The string to look for in triton-opt's output to confirm the bug.
    #
    # Use the assertion expression or the UNREACHABLE location, not a
    # generic marker: MLIR prints `error:` for every ill-typed module, and
    # delta debugging will otherwise "preserve" the crash by producing a
    # module that merely fails to verify.
    #   "Assertion `rank == 2' failed"
    #   "UNREACHABLE executed at"
    #   "LLVM ERROR:"
    bug_output = "Assertion `"

    reduced_src, reduced_flags = reduce_triton(
        testpath, triton_opt, flags, bug_output, env_prefix)

    version_result = subprocess.run(
        f"{triton_opt} --version", shell=True, capture_output=True, text=True)
    triton_version = (version_result.stdout or version_result.stderr or "").strip()

    src_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "triton-src")
    commit_result = subprocess.run(
        f"cd {src_dir} && git rev-parse HEAD",
        shell=True, capture_output=True, text=True)
    commit = (commit_result.stdout or "").strip() or "unknown"

    reproduce_cmd = (f"triton-opt {' '.join(reduced_flags)} "
                     f"./{os.path.basename(testpath)}")

    report_template = """
The following MLIR module:

```mlir
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

triton-opt version:
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
        version=triton_version,
        commit=commit,
        build_config=("cmake -G Ninja -DCMAKE_BUILD_TYPE=Release "
                      "-DTRITON_BUILD_PYTHON_MODULE=OFF -DTRITON_BUILD_UT=OFF "
                      "-DTRITON_CODEGEN_BACKENDS=\"nvidia;amd\" "
                      "(against Triton's prebuilt LLVM, which is configured "
                      "with LLVM_ENABLE_ASSERTIONS=ON)"),
        os_desc="Ubuntu 22.04, Docker fusion-fuzz-triton:latest",
    )

    print('\033[94m' + bug_report + '\033[0m')
