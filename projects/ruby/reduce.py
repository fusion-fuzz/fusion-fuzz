"""
projects/ruby/reduce.py — shrink a Ruby reproducer and print a report.

A fused program is two whole scripts stitched together; the assertion
failure usually needs a handful of their lines. Ruby's bug tracker will
not act on a 50-line reproducer that also depends on the fuzzer's own
assertion shim, so this narrows the program, then narrows the
interpreter flags, then prints a report ready to paste into
https://bugs.ruby-lang.org.

Same pipeline as the other adapters' reducers (see
projects/cpython/reduce.py): run_test / minimize_testcase /
further_minimize_testcase / reduce_flags, driven by the constants at the
bottom of this file.

Two things here differ from the CPython shape, both forced by the corpus.

**The shim goes first.** Seeds lifted out of test/ruby call test-unit
assertions that projects/ruby/ffl_shim.rb supplies, so the first
reduction step is to inline the handful the program actually calls as
no-op methods. A reproducer that needs `-rffl_shim` cannot be filed.

**Blocks are not line-deletable.** Deleting the `end` of a `do ... end`
leaves a program that fails to parse, which reads as "no longer
reproduces" and stops the reduction early. So a deletion that makes the
file unparseable (`ruby -c`) is rejected before it is even run — this is
the analogue of the Go reducer's unused-import rule.
"""

import os
import re
import subprocess

stdouterr = None

SHIM_STUB = """# minimal stand-ins for the test-unit assertions the original called
def assert_equal(*a) = nil
def assert_raise(*a) = (yield rescue nil)
def assert_nothing_raised(*a) = (yield rescue nil)
def assert_nil(*a) = nil
def assert(*a) = nil
def assert_predicate(*a) = nil
def assert_include(*a) = nil
def assert_match(*a) = nil
def assert_not_equal(*a) = nil
def assert_instance_of(*a) = nil
def assert_kind_of(*a) = nil
def assert_same(*a) = nil
def assert_operator(*a) = nil
def assert_raise_with_message(*a) = (yield rescue nil)
def assert_warning(*a) = (yield rescue nil)
def assert_syntax_error(*a) = nil
def assert_valid_syntax(*a) = nil
def omit(*a) = nil
def pend(*a) = nil
"""


def parses(testpath, ruby):
    """True when `ruby -c` accepts the file. A deletion that breaks the
    parse is not a smaller reproducer, it is a different program."""
    try:
        r = subprocess.run(f"{ruby} --disable-gems -c {testpath}", shell=True,
                           capture_output=True, text=True, timeout=30)
        return r.returncode == 0
    except Exception:
        return False


def run_test(cmd, bug_output, timeout=30):
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


def minimize_testcase(lines, bug_output, testpath, reproduce_cmd, ruby):
    """Delete one line at a time, from the end, keeping every deletion
    that still parses and still reproduces."""
    i = len(lines) - 1
    while i >= 0:
        candidate = lines[:i] + lines[i + 1:]
        _write(testpath, candidate)
        if parses(testpath, ruby) and run_test(reproduce_cmd, bug_output):
            lines = candidate
            print(f"  removed line {i + 1}, {len(lines)} left")
        i -= 1
    _write(testpath, lines)
    return lines


def further_minimize_testcase(lines, bug_output, testpath, reproduce_cmd, ruby):
    """Delete contiguous runs, largest first. Line-at-a-time cannot
    remove a `do ... end` block, because neither half of it can go
    alone."""
    size = max(len(lines) // 2, 1)
    while size >= 1:
        i = 0
        while i + size <= len(lines):
            candidate = lines[:i] + lines[i + size:]
            _write(testpath, candidate)
            if candidate and parses(testpath, ruby) and run_test(reproduce_cmd, bug_output):
                lines = candidate
                print(f"  removed {size} lines at {i + 1}, {len(lines)} left")
            else:
                i += 1
        size //= 2
    _write(testpath, lines)
    return lines


def reduce_flags(flags, bug_output, testpath, ruby):
    """Drop interpreter flags one at a time; keep the ones the crash
    needs (`--parser=prism` and `--enable-frozen-string-literal` really
    do decide some of these)."""
    kept = list(flags)
    for flag in list(kept):
        trial = [f for f in kept if f != flag]
        if run_test(f"{ruby} {' '.join(trial)} {testpath}", bug_output):
            kept = trial
            print(f"  dropped flag {flag}")
    return kept


def inline_shim(lines, bug_output, testpath, reproduce_cmd, ruby, shim_cmd):
    """Replace `-rffl_shim` with inlined no-op assertions. Returns the
    new (lines, cmd) when the crash survives, the originals when not."""
    candidate = SHIM_STUB.splitlines() + lines
    _write(testpath, candidate)
    if parses(testpath, ruby) and run_test(reproduce_cmd, bug_output):
        print("  shim inlined; reproducer no longer needs -rffl_shim")
        return candidate, reproduce_cmd
    _write(testpath, lines)
    return lines, shim_cmd


def reduce_ruby(testpath, ruby, flags, bug_output, shim_dir):
    with open(testpath) as f:
        lines = f.read().splitlines()

    shim_cmd = f"{ruby} {' '.join(flags)} -I {shim_dir} -rffl_shim {testpath}"
    plain_cmd = f"{ruby} {' '.join(flags)} {testpath}"
    if not run_test(shim_cmd, bug_output):
        print("The bug does not reproduce as given; check ruby path, flags and bug_output.")
        return lines, flags, shim_cmd

    lines, cmd = inline_shim(lines, bug_output, testpath, plain_cmd, ruby, shim_cmd)
    print(f"Reducing {len(lines)} lines ...")
    lines = minimize_testcase(lines, bug_output, testpath, cmd, ruby)
    lines = further_minimize_testcase(lines, bug_output, testpath, cmd, ruby)
    lines = minimize_testcase(lines, bug_output, testpath, cmd, ruby)
    kept_flags = reduce_flags(flags, bug_output, testpath,
                              ruby if "ffl_shim" not in cmd else f"{ruby} -I {shim_dir} -rffl_shim")
    return lines, kept_flags, cmd


if __name__ == "__main__":
    root = "/home/fuzz/WorkSpace/fusion-fuzz"
    testpath = "/tmp/ffl_repro.rb"
    ruby = f"{root}/projects/ruby/install/bin/ruby --disable-gems"
    shim_dir = f"{root}/projects/ruby"

    # Flags the crash was found under, from the bundle's test.sh.
    flags = ["--parser=parse.y"]

    # A distinctive fragment of the report: the assertion text, the
    # `[BUG]` message, or the sanitizer summary line.
    bug_output = "Assertion Failed"

    lines, kept_flags, cmd = reduce_ruby(testpath, ruby, flags, bug_output, shim_dir)

    poc = "\n".join(lines)
    commit = subprocess.run(f"cd {root}/projects/ruby/ruby-src && git rev-parse HEAD",
                            shell=True, capture_output=True, text=True).stdout.strip()
    build_config = ('../configure --prefix=... CC=clang-18 '
                    'cflags="-fsanitize=address -fno-omit-frame-pointer -DUSE_MN_THREADS=0" '
                    'cppflags="-DRUBY_DEBUG=1" optflags="-O1"')
    report = f"""
The following program:

```ruby
{poc}
```

makes the interpreter report:

```
{(stdouterr or '')[:3000]}
```

To reproduce:

```
{cmd}
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
Ubuntu 24.04 host, Docker ffe-ruby:latest
```

*This bug was found by [fusion-fuzz](https://github.com/fusion-fuzz/fusion-fuzz)*
"""
    print(report)
    with open("/tmp/ffl_ruby_report.md", "w") as f:
        f.write(report)
    print("report written to /tmp/ffl_ruby_report.md")
