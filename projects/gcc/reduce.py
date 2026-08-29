"""
projects/gcc/reduce.py — shrink a GCC ICE reproducer to something reportable.

A fused program is two whole source files stitched together; the ICE usually
needs a handful of their lines. GCC maintainers will not act on a 460-line
reproducer, so this narrows it before the bug is filed.

The oracle is the ICE *signature*, not merely "the compiler failed": a
reduction that trades the original ICE for a different one has found a
different bug and must be rejected, or the report ends up describing code
that no longer triggers what was observed.

Usage:
    python3 projects/gcc/reduce.py <file.c> [-- <compiler flags>]
"""

import os
import re
import subprocess
import sys
import tempfile

GCC_INSTALL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gcc-install")
_SIG_RE = re.compile(r"internal compiler error:\s*([^\n]+)")


def _compile(path, flags, cxx=False):
    binary = os.path.join(GCC_INSTALL, "bin", "g++" if cxx else "gcc")
    proc = subprocess.run(
        ["bash", "-c", f"ulimit -v 4194304; ulimit -c 0; {binary} {flags} {path}"],
        capture_output=True, text=True, errors="replace", timeout=120)
    return (proc.stdout or "") + (proc.stderr or "")


def signature_of(output):
    m = _SIG_RE.search(output)
    return m.group(1).strip() if m else None


def still_ices(lines, flags, target, cxx, tmpdir, ext):
    fd, path = tempfile.mkstemp(suffix=ext, dir=tmpdir)
    with os.fdopen(fd, "w") as f:
        f.write("\n".join(lines))
    try:
        return signature_of(_compile(path, flags, cxx)) == target
    except subprocess.TimeoutExpired:
        # A reduction that makes the compiler hang is not the bug we started
        # with, and keeping it would stall every later round.
        return False
    finally:
        os.unlink(path)


def reduce_file(path, flags, cxx=False):
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()

    target = signature_of(_compile(path, flags, cxx))
    if not target:
        print("Input does not ICE with these flags; nothing to reduce.")
        return None
    print(f"Target signature: {target}")
    print(f"Starting at {len(lines)} lines")

    ext = os.path.splitext(path)[1] or (".cc" if cxx else ".c")
    tmpdir = tempfile.mkdtemp()
    try:
        # Delta debugging: try to drop large chunks first and halve the chunk
        # size when a pass stops helping. Line-granular removal alone would
        # need O(n^2) compiles on a 460-line input.
        chunk = max(1, len(lines) // 2)
        while chunk >= 1:
            i = 0
            while i < len(lines):
                candidate = lines[:i] + lines[i + chunk:]
                if candidate and still_ices(candidate, flags, target, cxx, tmpdir, ext):
                    lines = candidate
                    print(f"  {len(lines)} lines (dropped {chunk} at {i})")
                else:
                    i += chunk
            if chunk == 1:
                break
            chunk = max(1, chunk // 2)
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    out = path + ".reduced" + ext
    with open(out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Reduced to {len(lines)} lines -> {out}")
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    extra = " ".join(sys.argv[3:]) if len(sys.argv) > 3 and sys.argv[2] == "--" else "-c -o /dev/null -O1"
    reduce_file(src, extra, cxx=src.endswith((".cc", ".cpp", ".C", ".cxx")))
