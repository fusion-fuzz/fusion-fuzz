#!/usr/bin/env python3
"""
tools/prereduce.py — shrink a saved crash bundle without knowing its
language, using the bundle's own test.sh as the reproduce command.

Why this exists. Every adapter has a reduce.py that deletes one line at
a time, which is the right final pass and hopeless as a first one: the
largest bundles here are 3,000-8,000 lines of generated compiler test,
and a line-at-a-time pass over 5,129 lines is 5,129 compiles before it
has removed anything at all. So these bundles have sat unreduced. This
does the coarse work first — delete a half, then a quarter, then an
eighth — which is what actually removes 90% of a machine-generated file,
and it does it against `test.sh`, so it needs no per-language knowledge
and works on all sixteen projects.

    python3 tools/prereduce.py output/bugs/<project>/<bundle> [options]

      --signal TEXT   the string that means "still the same crash"
                      (default: the longest fragment of the README key
                      that the saved test.out actually contains)
      --budget SEC    stop after this long (default 1500, so a run stays
                      inside the one-hour cap with room to spare)
      --floor N       smallest chunk to try (default 1)
      --write         write the result back as the bundle's min.<ext>
                      (otherwise it only reports what it would achieve)

The bundle is copied to a scratch directory first and test.sh is run
there, so the original is never touched until --write.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time


def bundle_test_file(bundle):
    """The reproducer inside `bundle` (test.<ext>, not test.sh/test.out)."""
    for name in sorted(os.listdir(bundle)):
        if name.startswith("test.") and not name.endswith((".sh", ".out")):
            return name
    return None


# Text that appears in every crash of a given toolchain and therefore
# identifies nothing: a reducer accepting one of these would "reduce" the
# program to any other crash, or to none at all.
_GENERIC = (
    "build config", "assertions", "please submit a bug report",
    "stack dump", "clang version", "llvm version", "program arguments",
    "see instructions", "preprocessed source", "compiler returned",
    "goroutine", "runtime stack", "control frame information",
    "c level backtrace", "aborted", "core dumped", "segmentation fault",
    "traceback", "execution halted", "note: this is a", "no such file",
)


def _too_generic(part):
    low = part.lower()
    if any(g in low for g in _GENERIC):
        return True
    # A path from the run that produced the bundle: the temp directory is
    # gone and the reduced file lives somewhere else.
    return bool(re.search(r'/tmp/|\.fused/|/home/', part))


def pick_signal(bundle):
    """A literal string that means "still the same crash".

    The `**Signature:**` line in README.md is a *normalised key* — masked
    numbers, stripped paths, collapsed whitespace — so it is usually not
    present verbatim in the compiler's output. But the bundle also keeps
    the real output in test.out, so the key's longest literal fragment
    that actually occurs there is both distinctive and greppable. That
    is what this returns; when there is no README key, the most
    distinctive line of test.out is used instead.
    """
    out_path = os.path.join(bundle, "test.out")
    saved = (open(out_path, encoding="utf-8", errors="replace").read()
             if os.path.exists(out_path) else "")
    readme = os.path.join(bundle, "README.md")
    if os.path.exists(readme):
        m = re.search(r'\*\*Signature:\*\*\s*`(.+?)`(?=\s*&nbsp;|\s*$)',
                      open(readme, encoding="utf-8", errors="replace").read(), re.M)
        if m:
            sig = m.group(1)
            # Fragments of the key, longest first; keep the first that
            # the saved output contains. `ICE:`/`Assertion:`/`Stack
            # dump:` are the campaign's own prefixes, not the compiler's
            # words, so they are cut off; a `>` separates the frames of a
            # stack-dump key, and any one frame identifies the site.
            sig = re.sub(r'^(?:ICE|Assertion|Stack dump|SUMMARY|UBSAN|internal):\s*', '', sig)
            parts = re.split(r'<[^>]*>|\b0x\w*|\s{2,}|["`]|\s+>\s+', sig)
            parts = sorted((p.strip(" :,()[]{}<>-\"'`") for p in parts),
                           key=len, reverse=True)
            for part in parts:
                if len(part) < 12 or _too_generic(part):
                    continue
                if not saved or part in saved:
                    return part
    for line in saved.splitlines():
        line = line.strip()
        if re.search(r'Assertion|SUMMARY:|panic:|caught segfault|'
                     r'internal compiler error|\[BUG\]|UNREACHABLE|Debug failure', line):
            # Drop the leading path, which differs between runs.
            cut = re.sub(r'^.*?(?=Assertion|SUMMARY:|panic:|caught|internal compiler|'
                         r'\[BUG\]|UNREACHABLE|Debug failure)', '', line)[:160]
            if len(cut) >= 12 and not _too_generic(cut):
                return cut
    return None


def run_once(workdir, signal, timeout):
    try:
        r = subprocess.run(["bash", "./test.sh"], cwd=workdir, capture_output=True,
                           text=True, encoding="iso-8859-1", timeout=timeout)
    except Exception:
        return False
    return signal in (r.stdout or "") or signal in (r.stderr or "")


def chunk_reduce(lines, workdir, test_name, signal, budget, floor, timeout):
    """Delete runs of lines, largest first. Returns (lines, tests_run)."""
    started = time.time()
    tests = 0
    size = max(len(lines) // 2, 1)
    while size >= floor:
        i = 0
        while i + size <= len(lines):
            if time.time() - started > budget:
                print(f"  budget spent after {tests} tests")
                return lines, tests
            candidate = lines[:i] + lines[i + size:]
            with open(os.path.join(workdir, test_name), "w") as f:
                f.write("\n".join(candidate) + "\n")
            tests += 1
            if candidate and run_once(workdir, signal, timeout):
                lines = candidate
                print(f"  -{size} lines at {i + 1}: {len(lines)} left")
            else:
                i += size
        size //= 2
    with open(os.path.join(workdir, test_name), "w") as f:
        f.write("\n".join(lines) + "\n")
    return lines, tests


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle")
    ap.add_argument("--signal")
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--floor", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--from-min", action="store_true",
                    help="start from the bundle's min.<ext> when it is already "
                         "smaller than test.<ext> (a second, finer pass)")
    args = ap.parse_args()

    bundle = os.path.abspath(args.bundle)
    test_name = bundle_test_file(bundle)
    if not test_name:
        sys.exit(f"no test.<ext> in {bundle}")
    signal = args.signal or pick_signal(bundle)
    if not signal:
        sys.exit("no --signal given and none found in README.md")
    print(f"bundle {os.path.basename(bundle)}\n  file {test_name}\n  signal {signal!r}")

    workdir = tempfile.mkdtemp(prefix="ffl_prereduce_")
    try:
        for name in os.listdir(bundle):
            src = os.path.join(bundle, name)
            if os.path.isfile(src):
                shutil.copy2(src, workdir)
        if args.from_min:
            ext = test_name.split(".", 1)[1]
            min_path = os.path.join(bundle, f"min.{ext}")
            if os.path.exists(min_path):
                shutil.copy2(min_path, os.path.join(workdir, test_name))
                print("  starting from min." + ext)
        if not run_once(workdir, signal, args.timeout):
            sys.exit("the bundle does not reproduce as saved; pass --signal by hand")
        lines = open(os.path.join(workdir, test_name),
                     encoding="utf-8", errors="replace").read().splitlines()
        print(f"  reproduces; {len(lines)} lines")
        before = len(lines)
        lines, tests = chunk_reduce(lines, workdir, test_name, signal,
                                    args.budget, args.floor, args.timeout)
        print(f"{before} -> {len(lines)} lines in {tests} tests")
        if args.write:
            ext = test_name.split(".", 1)[1]
            out = os.path.join(bundle, f"min.{ext}")
            with open(out, "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"written to {out}")
        else:
            print("(not written; pass --write to update the bundle's min file)")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
