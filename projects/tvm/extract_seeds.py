"""
projects/tvm/extract_seeds.py — turn TVM's Python tests into standalone
TVMScript seeds.

A TVM test defines its modules inline, usually inside a test function:

    def test_x():
        @I.ir_module(s_tir=True)
        class Before:
            @T.prim_func
            def main(A: T.Buffer((16,), "float32")): ...

Each decorated `@I.ir_module ... class` (any indentation) becomes one
seed: the block is dedented to column 0, renamed `Module`, and prefixed
with the standard script imports. Standalone `@T.prim_func` /
`@R.function` defs outside a class are wrapped into a module of their
own. Blocks that reference names from the enclosing test (a `size`
variable, a helper) fail alone and are dropped by pre-analysis.
"""

import os
import re
import sys
import textwrap

HEADER = (
    "import tvm\n"
    "from tvm.script import ir as I\n"
    "from tvm.script import tirx as T\n"
    "from tvm.script import relax as R\n\n"
)
_DECOR_RE = re.compile(r'^(\s*)@I\.ir_module\b.*$')
_FUNC_DECOR_RE = re.compile(r'^(\s*)@(?:T\.prim_func|R\.function)\b.*$')
_MODULE_ALIAS_RE = re.compile(r'\bclass\s+([A-Za-z_]\w*)\s*(\([^)]*\))?\s*:')


def _block(lines, start, indent):
    """Lines from `start` (a decorator line at `indent`) through the end of
    the indented body that follows the header."""
    out = [lines[start]]
    i = start + 1
    # header line(s): more decorators, then the class/def header
    while i < len(lines) and (lines[i].strip().startswith("@") or not lines[i].strip()):
        out.append(lines[i]); i += 1
    if i >= len(lines):
        return None, i
    out.append(lines[i]); i += 1              # class/def header
    while i < len(lines):
        ln = lines[i]
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent:
            break
        out.append(ln); i += 1
    while out and not out[-1].strip():
        out.pop()
    return out, i


def extract(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        src = f.read()
    lines = src.splitlines()
    seeds, i = [], 0
    while i < len(lines):
        m = _DECOR_RE.match(lines[i])
        if m:
            blk, nxt = _block(lines, i, len(m.group(1)))
            if blk:
                text = textwrap.dedent("\n".join(blk))
                text = _MODULE_ALIAS_RE.sub("class Module:", text, count=1)
                seeds.append(("module", text))
            i = nxt if nxt > i else i + 1
            continue
        m = _FUNC_DECOR_RE.match(lines[i])
        if m and len(m.group(1)) == 0:
            blk, nxt = _block(lines, i, 0)
            if blk:
                body = textwrap.indent(textwrap.dedent("\n".join(blk)), "    ")
                seeds.append(("func", "@I.ir_module\nclass Module:\n" + body))
            i = nxt if nxt > i else i + 1
            continue
        i += 1
    return seeds


def main(tests_root, out_dir, max_lines=400):
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for root, _dirs, files in os.walk(tests_root):
        for fn in sorted(files):
            if not fn.endswith(".py") or fn.startswith("conftest"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, tests_root).replace(os.sep, "__")[:-3]
            for k, (kind, text) in enumerate(extract(path)):
                if text.count("\n") > max_lines or "T.Buffer" not in text and "R.Tensor" not in text and "T.prim_func" not in text:
                    continue
                with open(os.path.join(out_dir, f"{rel}__{k}.py"), "w", encoding="utf-8") as f:
                    f.write(HEADER + text.rstrip() + "\n")
                n += 1
    return n


if __name__ == "__main__":
    print(main(sys.argv[1], sys.argv[2]))
