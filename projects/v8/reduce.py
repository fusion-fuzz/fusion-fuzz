import os
import subprocess

stdouterr = None

# Function to run the test command and check for bug presence
def run_test(cmd, bug_output, timeout=20):
    """
    Executes the provided command to run the JS test and checks
    if the expected bug output or any sanitizer error appears.
    """
    # Run the command and capture the output
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, encoding='iso-8859-1', timeout=timeout)
    except:
        return False


    # Check if the bug output or any sanitizer errors are in the stdout/stderr
    if not (bug_output in result.stdout or bug_output in result.stderr) and \
       ("LeakSanitizer" not in result.stdout and "LeakSanitizer" not in result.stderr):

        # If another sanitizer message shows up, print the error
        if "Sanitizer" in result.stdout or "Sanitizer" in result.stderr:
            print("Other error messages found:")
            print(result.stdout)
            print(result.stderr)
            # Uncomment below if you want to pause for input when this happens
            # input()

    if bug_output in result.stdout or bug_output in result.stderr:
        global stdouterr
        if stdouterr == None:
            stdouterr = result.stderr

    # Return True if the bug output is found in the test results
    return bug_output in result.stdout or bug_output in result.stderr

# Function to minimize the test case by removing lines
def minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    print("reducing .. it may cost some times")
    """
    Minimizes the test case by iteratively removing lines and checking
    if the bug still reproduces. Uses a stepwise approach for efficiency.
    """
    n = len(lines)
    step = max(n // 2, 1)  # Start with removing half of the lines at a time

    init_step = step

    # Reduce the number of lines step by step
    while step > 0:
        print(f"Current step: {step}")

        # Try removing 'step' lines at a time
        for i in range(0, n, step):
            temp_lines = lines[:i] + lines[i+step:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))

            # If the bug reproduces, accept this as the minimized version
            if run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break
        else:
            step //= 2  # If no further reduction is found, reduce step size

    return lines, init_step

# Function for further minimizing by removing multiple lines at a time
def further_minimize_testcase(lines, bug_output, testpath, reproduce_cmd):
    """
    Further minimizes the test case by removing 2 to 5 lines at a time
    and checking if the bug still reproduces.
    """
    n = len(lines)

    # Try removing 2 to 5 lines at a time
    for count in range(2, 6):
        # print(f"Trying to remove {count} lines at a time.")

        # Try removing 'count' lines from each part of the test case
        for i in range(n - count + 1):
            temp_lines = lines[:i] + lines[i+count:]
            with open(testpath, "w") as f:
                f.write("\n".join(temp_lines))

            # If the bug reproduces, accept this as the minimized version
            if run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output) or run_test(reproduce_cmd, bug_output):
                lines = temp_lines
                n = len(lines)
                break

    return lines

def reduce_flags(flags, bug_output, testpath, d8path, env_prefix=""):
    """Try removing interpreter flags one at a time.

    Same signature as every other adapter's reduce_flags (see
    projects/clang/reduce.py), so one triage procedure covers every
    language. The caller below still takes `config` as a single string —
    that is this family's convention and callers depend on it — and splits
    it here.
    """
    reduced = [f for f in flags if f]
    changed = True
    while changed:
        changed = False
        for i in range(len(reduced)):
            trial = reduced[:i] + reduced[i + 1:]
            cmd = f"{env_prefix}{d8path} {' '.join(trial)} {testpath}"
            if run_test(cmd, bug_output) or run_test(cmd, bug_output):
                reduced = trial
                changed = True
                break
    return reduced


def reduce_js(testpath, d8path, config, bug_output, env_prefix=""):
    reproduce_cmd = f'{env_prefix}{d8path} {config} {testpath}'
    # Initial test to verify if the reproduce command triggers the bug
    if not run_test(reproduce_cmd, bug_output) and not run_test(reproduce_cmd, bug_output) and not run_test(reproduce_cmd, bug_output):
        return "bug not reproduced when reducing", "bug not reproduced when reducing"
    else:
        while True:
            # Read the original test file lines
            with open(testpath, "r") as f:
                lines = f.readlines()

            # Strip any extra whitespace or newlines
            lines = [line.strip() for line in lines]

            # Begin minimizing the test case by removing lines
            minimized_lines, init_step = minimize_testcase(lines, bug_output, testpath, reproduce_cmd)

            # Further minimize by removing multiple lines at once
            further_minimized_lines = further_minimize_testcase(minimized_lines, bug_output, testpath, reproduce_cmd)

            # Restore the original test case in the file
            with open(testpath, "w") as f:
                f.write("\n".join(further_minimized_lines))

            n = len(further_minimized_lines)
            step = max(n // 2, 1)
            if step==init_step:
                print("reducing js finished")
                break
        reducedjs = "\n".join(further_minimized_lines)

        reduced_config = " ".join(
            reduce_flags(config.split(), bug_output, testpath, d8path, env_prefix))

        return reducedjs, reduced_config.strip('\n')



if __name__ == "__main__":

    # Path to the reproducer. d8 takes a plain script path, so unlike the
    # CPython reducer there is no stdlib-shadowing hazard in the name —
    # but the file still has to live outside the checkout, because a fused
    # script can and does delete things in its working directory.
    testpath = "/tmp/ffl_repro.js"

    # default d8 path
    d8path = "/home/fuzz/WorkSpace/fusion-fuzz/projects/v8/v8-src/v8/out/fuzz/d8"

    # The runtime flags the crash was found under. Copy them from the
    # `command` line of the finding — for V8 this is the *most* important
    # field to get right, because a bug that needs --stress-maglev or
    # --deopt-every-n-times will not reproduce at all without it, and the
    # config reduction below will then strip every flag as "unnecessary".
    config = '--allow-natives-syntax --fuzzing'

    # The expected bug output that we are trying to reproduce.
    # if sanitizers' alerts
    bug_output = 'Sanitizer'
    # if a DCHECK / CHECK failure
    #bug_output = 'Debug check failed'
    # if a sandbox violation
    #bug_output = 'V8 sandbox violation detected'

    reducedjs, reduced_config = reduce_js(testpath, d8path, config, bug_output)

    reduced_config = f'{d8path} {reduced_config} ./ffl_repro.js'

    # auto generate bug report
    report_template = "\nThe following code:\n\n```javascript\n{poc}\n```\n\nResulted in this output:\n```\n{stdouterr}\n```\n\nTo reproduce:\n```\n{config}\n```\n\nCommit:\n```\n{commit}\n```\n\nBuild configuration:\n```\n{build_config}\n```\n\nOperating System:\n```\n{os}\n```\n\n*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*\n"

    os.system("cd /home/fuzz/WorkSpace/fusion-fuzz/projects/v8/v8-src/v8/; git rev-parse HEAD > /tmp/v8_commit")
    f = open("/tmp/v8_commit","r")
    commit = f.read()
    f.close()

    build_config = "gn gen out/fuzz --args='is_debug = true v8_enable_slow_dchecks = true v8_enable_verify_heap = true v8_enable_verify_csa = true v8_enable_object_print = true is_asan = true'"

    # NOTE: this shadows the `os` module imported at the top. It only works
    # because the os.system() call above has already run; keep it last.
    os = "Ubuntu 24.04 Docker, fusion-fuzz-v8:latest"

    bug_report = report_template.format(
        poc = reducedjs,
        stdouterr = stdouterr,
        config = reduced_config,
        commit = commit,
        build_config = build_config,
        os = os
    )

    print('\033[94m'+bug_report+'\033[0m')
