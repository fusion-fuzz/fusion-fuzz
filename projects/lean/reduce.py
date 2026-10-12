"""
projects/lean/reduce.py — shrink a Lean reproducer while it keeps failing
the same way.

    python3 projects/lean/reduce.py [--clean] FILE PATTERN [lean flags...]

PATTERN is a regex searched in lean's combined output (e.g. the oracle
line: "PANIC at Option.get", "\\(kernel\\) declaration type mismatch").
The reduction is grammar-aware rather than line-based: it first deletes
whole commands (core/lean_text.py's split, delta-debugging style with
halving chunks), then header imports, then single lines, so the result
stays a sequence of well-formed commands for as long as possible — a
line- or token-level pass alone turns a well-formed program into an
ill-formed one and quietly changes the bug into an error-recovery one.
With --clean, a candidate also has to elaborate without any ordinary
error (only the oracle's own messages may be errors): the reproducer
then shows a bug on *valid* input, not on Lean's error-recovery path.
The result is printed and written to FILE.min.lean.
"""

import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from core import lean_text as LT  # noqa: E402

_LEAN = os.environ.get("FFL_LEAN_BIN") or os.path.join(
    _ROOT, "projects", "lean", "lean4", "build", "relwithassert", "stage1", "bin", "lean")


_ORACLE_LINE = re.compile(r"\(kernel\)|unknown free variable|internal error|Internal error|"
                          r"LCNF|IR check failed|\(interpreter\)|PANIC")
_PLAIN_ERROR = re.compile(r":\d+:\d+: error(?:\([\w.]+\))?: ")


def _oracle(pat, flags, tmp, timeout=90, clean=False):
    rx = re.compile(pat, re.M)

    def bad(text):
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        try:
            r = subprocess.run([_LEAN, *flags, tmp], capture_output=True, text=True,
                               stdin=subprocess.DEVNULL, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        out = r.stdout + r.stderr
        if clean and any(_PLAIN_ERROR.search(ln) and not _ORACLE_LINE.search(ln)
                         for ln in out.split("\n")):
            return False
        return rx.search(out) is not None
    return bad


def _ddmin(units, test):
    """Classic ddmin over a list: returns a 1-minimal sublist."""
    n = 2
    while len(units) >= 2:
        chunk = max(1, len(units) // n)
        for i in range(0, len(units), chunk):
            cand = units[:i] + units[i + chunk:]
            if cand and test(cand):
                units, n = cand, max(n - 1, 2)
                break
        else:
            if chunk == 1:
                break
            n = min(len(units), n * 2)
    return units


def reduce(path, pat, flags, clean=False):
    tmp = path + ".try.lean"
    bad = _oracle(pat, flags, tmp, clean=clean)
    src = open(path, encoding="utf-8").read()
    if not bad(src):
        raise SystemExit("does not reproduce")
    header, body = LT.split_header(src)
    lines = body.split("\n")
    cmds = LT.split_commands(body)
    pre = lines[:cmds[0].start] if cmds else lines
    units = [lines[c.start:c.end] for c in cmds]

    def text(hdr, us):
        return "\n".join(list(hdr) + pre + [ln for u in us for ln in u]) + "\n"

    units = _ddmin(units, lambda us: bad(text(header, us)))
    header = _ddmin(list(header) + [None], lambda hs: bad(text([h for h in hs if h], units)))
    header = [h for h in header if h]
    flat = [ln for u in units for ln in u]
    pre[:] = []
    flat = _ddmin(flat, lambda ls: bad("\n".join(header + ls) + "\n"))
    out = "\n".join(header + flat) + "\n"
    with open(path + ".min.lean", "w", encoding="utf-8") as f:
        f.write(out)
    os.remove(tmp)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    clean = args[:1] == ["--clean"]
    if clean:
        args = args[1:]
    if len(args) < 2:
        raise SystemExit(__doc__)
    print(reduce(args[0], args[1], args[2:], clean=clean))
