import os
import subprocess

stdouterr = None

def run_test(cmd, bug_output):
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
    except:
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
            temp_lines = lines[:i] + lines[i+step:]
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
            temp_lines = lines[:i] + lines[i+count:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))
            if run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output) or \
               run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
    return lines


def reduce_flags(flags, bug_output, testpath, rustc_path):
    """Try removing rustc flags one at a time."""
    reduced = flags[:]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i+1:]
            cmd = f"{rustc_path} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_rust(testpath, rustc_path, flags, bug_output):
    reproduce_cmd = f"{rustc_path} {' '.join(flags)} {testpath}"

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
            print("Reducing Rust source finished.")
            break

    reduced_rs = "\n".join(further_minimized_lines)

    print("Reducing flags...")
    reduced_flags = reduce_flags(flags, bug_output, testpath, rustc_path)
    print(f"Reduced flags: {reduced_flags}")

    return reduced_rs, reduced_flags


if __name__ == "__main__":
    testpath = "/tmp/test.rs"
    rustc_path = "/usr/local/cargo/bin/rustc"

    # Add any extra rustc flags here, e.g. ["--edition", "2021"]
    flags = ["--edition=2018", "-C", "opt-level=z", "-C", "codegen-units=16", "-C",  "debug-assertions=yes"]

    # The string to look for in rustc's output to confirm the bug
    # Examples:
    #   "internal compiler error"
    #   "Should not have unglued last token"
    #   "expanded a dummy bang macro"
    #   "unwrap() on a `None`"
    bug_output = "internal compiler error: Could not resolve Lifetime"

    reduced_rs, reduced_flags = reduce_rust(testpath, rustc_path, flags, bug_output)

    # Get rustc version
    version_result = subprocess.run(
        f"{rustc_path} --version", shell=True, capture_output=True, text=True)
    rustc_version = version_result.stdout.strip()

    reproduce_cmd = f"{rustc_path} {' '.join(reduced_flags)} ./test.rs"

    report_template = """
The following code:

```rust
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
        poc=reduced_rs,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=rustc_version,
    )

    print('\033[94m' + bug_report + '\033[0m')
