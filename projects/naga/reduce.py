import os
import subprocess

stdouterr = None


def run_test(cmd, bug_output):
    """
    Execute the provided Naga command and check whether the expected bug
    output appears in stdout/stderr.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=15,
        )
    except Exception:
        return False

    combined = result.stdout + result.stderr

    if bug_output not in combined:
        if "panicked at" in combined or "Sanitizer" in combined:
            print("Other crash-like output found:")
            print(result.stdout)
            print(result.stderr)

    if bug_output in combined:
        global stdouterr
        if stdouterr is None:
            stdouterr = result.stderr if result.stderr else result.stdout

    return bug_output in combined


def minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    print("reducing .. it may cost some times")
    n = len(lines)
    step = max(n // 2, 1)
    init_step = step

    while step > 0:
        print(f"Current step: {step}")
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


def reduce_backend_args(backend_args, bug_output, testpath, nagapath, env_prefix):
    """
    Try removing optional Naga backend/output arguments. The input path is
    always preserved; backend_args is usually empty or an output file/profile
    suffix copied from the original crash command.
    """
    reduced = backend_args.split()
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            test_cmd = f"{env_prefix}{nagapath} {testpath} {' '.join(trial)}"
            if run_test(test_cmd, bug_output) or \
               run_test(test_cmd, bug_output) or \
               run_test(test_cmd, bug_output):
                reduced = trial
                changed = True
                break
    return " ".join(reduced)


def reduce_naga(testpath, nagapath, backend_args, bug_output, env_prefix=""):
    reproduce_cmd = f"{env_prefix}{nagapath} {testpath} /tmp/test.hlsl {backend_args}".strip()

    if not run_test(reproduce_cmd, bug_output) and \
       not run_test(reproduce_cmd, bug_output) and \
       not run_test(reproduce_cmd, bug_output):
        return "bug not reproduced when reducing", "bug not reproduced when reducing"

    while True:
        with open(testpath, "r") as f:
            lines = f.readlines()

        lines = [line.rstrip("\n") for line in lines]

        minimized_lines, init_step = minimize_testcase(
            lines, bug_output, testpath, reproduce_cmd)

        further_minimized_lines = further_minimize_testcase(
            minimized_lines, bug_output, testpath, reproduce_cmd)

        with open(testpath, "w") as f:
            f.write("\n".join(further_minimized_lines))

        n = len(further_minimized_lines)
        step = max(n // 2, 1)
        if step == init_step:
            print("reducing naga finished")
            break

    reduced_wgsl = "\n".join(further_minimized_lines)

    print("reducing naga backend args")
    reduced_backend_args = reduce_backend_args(
        backend_args, bug_output, testpath, nagapath, env_prefix)

    return reduced_wgsl, reduced_backend_args.strip("\n")


if __name__ == "__main__":

    # Copy the crashing WGSL file to /tmp/test.wgsl before running, or
    # point this directly at a bug bundle's test.wgsl.
    testpath = "/tmp/test.wgsl"

    # Default naga-cli path. If cargo installed it elsewhere, set the full
    # path here, e.g. /home/fuzz/.cargo/bin/naga.
    nagapath = "naga"

    # Optional backend/output args copied from the crash command.
    # Examples:
    #   ""
    #   "/tmp/test.spv"
    #   "/tmp/test.metal"
    #   "/tmp/test.vert --profile es310"
    backend_args = ""

    env_prefix = (
        "RUST_BACKTRACE=1 "
        "ASAN_OPTIONS='abort_on_error=1:detect_leaks=0:symbolize=1' "
        "UBSAN_OPTIONS='print_stacktrace=1:halt_on_error=1' "
    )

    # The expected bug output that we are trying to reproduce.
    # Examples:
    #   "panicked at"
    #   "thread 'main' panicked"
    #   "Segmentation fault"
    #   "AddressSanitizer"
    bug_output = "panicked at"

    reducedwgsl, reduced_backend_args = reduce_naga(
        testpath, nagapath, backend_args, bug_output, env_prefix)

    reproduce_cmd = f"{env_prefix}{nagapath} /tmp/test.wgsl /tmp/test.hlsl {reduced_backend_args}".strip()

    version_result = subprocess.run(
        f"{nagapath} --version", shell=True, capture_output=True,
        text=True, errors="replace")
    naga_version = version_result.stdout.strip() or version_result.stderr.strip()

    project_root = os.path.dirname(os.path.abspath(__file__))
    try:
        commit_result = subprocess.run(
            "git rev-parse HEAD",
            shell=True,
            cwd=os.path.join(project_root, "wgpu"),
            capture_output=True,
            text=True,
            errors="replace",
        )
        commit = commit_result.stdout.strip()
    except Exception:
        commit = "unknown"

    os_info = "Ubuntu Host, Naga built from gfx-rs/wgpu trunk"

    report_template = "\nThe following code:\n\n```wgsl\n{poc}\n```\n\nResulted in this output:\n```\n{stdouterr}\n```\n\nTo reproduce:\n```\n{cmd}\n```\n\nNaga version:\n```\n{version}\n```\n\nCommit:\n```\n{commit}\n```\n\nOperating System:\n```\n{os}\n```\n\n*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*\n"

    bug_report = report_template.format(
        poc=reducedwgsl,
        stdouterr=stdouterr,
        cmd=reproduce_cmd,
        version=naga_version,
        commit=commit,
        os=os_info,
    )

    print('\033[94m' + bug_report + '\033[0m')
