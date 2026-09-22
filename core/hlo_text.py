"""
core/hlo_text.py — a small, faithful-enough model of XLA's HLO text format.

Shared by the XLA fusion strategies in core/fusion.py and the seed analysis
in projects/xla/analyzer.py, so both read a module the same way.

What the format looks like (xla/hlo/parser/hlo_parser.cc is the authority):

    HloModule name, is_scheduled=true, entry_computation_layout={...}

    %reducer (a: f32[], b: f32[]) -> f32[] {          // a computation
      %a = f32[] parameter(0)
      %b = f32[] parameter(1)
      ROOT %add = f32[] add(f32[] %a, f32[] %b)
    }

    ENTRY %main (p: f32[8]) -> f32[] {                 // the entry
      %p = f32[8] parameter(0)
      %c = f32[] constant(0)
      ROOT %r = f32[] reduce(%p, %c), dimensions={0}, to_apply=%reducer
    }

An instruction is `[ROOT] name = shape opcode(operands)[, attr=value ...]`.
Names may carry a leading `%` or not; operands may carry their shape in
front of the name or not; a computation may declare a signature
`(params) -> result` or not (the parser infers it from the body when it is
absent). Instruction names are scoped to their computation, computation
names to the module. Layout suffixes (`{3,2,1,0}`) are part of a shape's
text but not of its type, so shapes are compared with them stripped.

This model keeps every instruction's pieces (name, shape, opcode, operand
tokens, attribute tail) so a strategy can rename names, renumber
parameters and re-emit the text; it does not interpret attributes beyond
finding the computation names they reference.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lexical pieces
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r'^\s*HloModule\b(.*)$')
_COMP_OPEN_RE = re.compile(
    r'^\s*(ENTRY\s+)?(%?)([\w.\-]+)\s*(\(.*?\)\s*->\s*.+?)?\s*\{\s*$')
_COMP_CLOSE_RE = re.compile(r'^\s*\}\s*(//.*)?$')
_INSTR_HEAD_RE = re.compile(r'^\s*(ROOT\s+)?(%?)([\w.\-]+)\s*=\s*(.*)$', re.S)
# A line *starts* an instruction only when what follows `=` is a shape: a
# tuple `(`, or `type[` (`f32[8]`, `pred[]`, `token[]`, `bf16[2,3]`). A
# wrapped attribute line (`rhs_contracting_dims={1},`, `to_apply=%r`) also
# matches `name = ...` and must stay a continuation of the line above.
_INSTR_START_RE = re.compile(r'^\s*(ROOT\s+)?%?[\w.\-]+\s*=\s*(\(|[a-z][a-z0-9]*\[)')
_OPCODE_RE = re.compile(r'([a-z][\w\-]*)\(')
_COMMENT_RE = re.compile(r'//.*?$|/\*.*?\*/', re.M | re.S)
_LAYOUT_RE = re.compile(r'\{[^{}]*\}')
_NAME_RE = re.compile(r'%?([\w.\-]+)$')

# Attribute keys whose value names one or more computations.
COMPUTATION_ATTRS = (
    "to_apply", "calls", "condition", "body", "branch_computations",
    "called_computations", "select", "scatter", "update_computation",
    "async_execution_thread_computation", "computation",
)
_COMP_REF_RE = re.compile(
    r'\b(' + "|".join(COMPUTATION_ATTRS) + r')\s*=\s*(\{[^{}]*\}|%?[\w.\-]+)')
_CTRL_PRED_RE = re.compile(r'control-predecessors\s*=\s*\{([^{}]*)\}')


@dataclass
class HloInstr:
    name: str                     # without the leading %
    shape: str                    # as written (layouts included)
    opcode: str
    operands: List[str]           # raw operand tokens, e.g. "f32[2] %p" or "%p"
    attrs: str                    # everything after the operand list, "" or ", k=v, ..."
    is_root: bool = False
    comment: str = ""             # trailing // comment, if any

    # -- derived -----------------------------------------------------------
    def operand_names(self) -> List[str]:
        return [n for n in (operand_name(t) for t in self.operands) if n]

    def type_shape(self) -> str:
        return normalize_shape(self.shape)

    def param_index(self) -> Optional[int]:
        if self.opcode == "parameter" and self.operands:
            try:
                return int(self.operands[0].strip())
            except ValueError:
                return None
        return None

    def computation_refs(self) -> List[str]:
        out = []
        for _key, val in _COMP_REF_RE.findall(self.attrs):
            if val.startswith("{"):
                out.extend(n.lstrip("%") for n in re.findall(r'%?([\w.\-]+)', val[1:-1]))
            else:
                out.append(val.lstrip("%"))
        return out

    def render(self) -> str:
        root = "ROOT " if self.is_root else ""
        ops = ", ".join(self.operands)
        text = f"  {root}%{self.name} = {self.shape} {self.opcode}({ops}){self.attrs}"
        if self.comment:
            text += f"  {self.comment}"
        return text


@dataclass
class HloComputation:
    name: str
    is_entry: bool
    signature: Optional[str]      # "(p: f32[8]) -> f32[]" or None
    instrs: List[HloInstr] = field(default_factory=list)
    leading: List[str] = field(default_factory=list)   # comment/blank lines before it

    def root(self) -> Optional[HloInstr]:
        for i in self.instrs:
            if i.is_root:
                return i
        return self.instrs[-1] if self.instrs else None

    def params(self) -> List[HloInstr]:
        ps = [i for i in self.instrs if i.opcode == "parameter"]
        return sorted(ps, key=lambda i: i.param_index() if i.param_index() is not None else 1 << 30)

    def names(self) -> List[str]:
        return [i.name for i in self.instrs]

    def render(self, with_signature=True) -> str:
        head = "ENTRY " if self.is_entry else ""
        sig = f" {self.signature}" if (with_signature and self.signature) else ""
        lines = list(self.leading)
        lines.append(f"{head}%{self.name}{sig} {{")
        lines.extend(i.render() for i in self.instrs)
        lines.append("}")
        return "\n".join(lines)


@dataclass
class HloModule:
    name: str
    header_attrs: str             # text after "HloModule <name>" ("" or ", k=v...")
    computations: List[HloComputation] = field(default_factory=list)
    preamble: List[str] = field(default_factory=list)   # lines before the header
    had_header: bool = True

    def entry(self) -> Optional[HloComputation]:
        for c in self.computations:
            if c.is_entry:
                return c
        # A module with a single computation and no ENTRY keyword: that
        # computation is the entry (the parser accepts this form).
        if len(self.computations) == 1:
            return self.computations[0]
        return self.computations[-1] if self.computations else None

    def non_entry(self) -> List[HloComputation]:
        e = self.entry()
        return [c for c in self.computations if c is not e]

    def get(self, name: str) -> Optional[HloComputation]:
        for c in self.computations:
            if c.name == name:
                return c
        return None

    def render(self, with_signatures=True) -> str:
        out = list(self.preamble)
        out.append(f"HloModule {self.name}{self.header_attrs}")
        for c in self.computations:
            out.append("")
            out.append(c.render(with_signatures))
        return "\n".join(out).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _balanced_end(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index of the close_ch matching the open_ch at text[start], or -1."""
    depth = 0
    i = start
    in_str = False
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == '\\':
                i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level(text: str, sep: str = ",") -> List[str]:
    """Split at `sep` outside (), [], {} and strings."""
    out, depth, cur, in_str = [], 0, [], False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_str:
            cur.append(ch)
            if ch == '\\' and i + 1 < len(text):
                cur.append(text[i + 1]); i += 1
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True; cur.append(ch)
        elif ch in "([{":
            depth += 1; cur.append(ch)
        elif ch in ")]}":
            depth -= 1; cur.append(ch)
        elif ch == sep and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
        i += 1
    if cur or out:
        out.append("".join(cur))
    return [s.strip() for s in out if s.strip()]


def operand_name(token: str) -> Optional[str]:
    """`f32[2]{0} %p` -> "p"; `%p` -> "p"; `p` -> "p"; a literal -> None."""
    tok = token.strip()
    if not tok:
        return None
    m = _NAME_RE.search(tok)
    if not m:
        return None
    name = m.group(1)
    # A bare number is a parameter index or a constant operand, not a name.
    if re.fullmatch(r'-?\d+(\.\d+)?', name) and not tok.startswith("%"):
        return None
    return name


def normalize_shape(shape: str) -> str:
    """Shape text without layouts or whitespace: the comparable type."""
    s = _LAYOUT_RE.sub("", shape)
    return re.sub(r'\s+', '', s)


def parse_instruction(text: str) -> Optional[HloInstr]:
    """One instruction (continuation lines already joined), or None."""
    comment = ""
    # A trailing line comment; keep it, a fusion tag lands there.
    cm = re.search(r'\s*//[^\n]*$', text)
    if cm and '"' not in text[cm.start():]:
        comment = text[cm.start():].strip()
        text = text[:cm.start()]
    m = _INSTR_HEAD_RE.match(text.strip())
    if not m:
        return None
    is_root, name, rest = bool(m.group(1)), m.group(3), m.group(4).strip()
    # The opcode is the first `word(` whose preceding text (the shape) is
    # bracket-balanced.
    for om in _OPCODE_RE.finditer(rest):
        shape = rest[:om.start()].strip()
        if shape.count("(") != shape.count(")") or shape.count("[") != shape.count("]") \
                or shape.count("{") != shape.count("}"):
            continue
        if not shape:
            continue
        opcode = om.group(1)
        open_i = om.end() - 1
        close_i = _balanced_end(rest, open_i, "(", ")")
        if close_i == -1:
            return None
        operands = split_top_level(rest[open_i + 1:close_i])
        attrs = rest[close_i + 1:].rstrip()
        return HloInstr(name=name, shape=shape, opcode=opcode, operands=operands,
                        attrs=attrs, is_root=is_root, comment=comment)
    return None


def parse_module(text: str) -> Optional[HloModule]:
    """Parse HLO text. Returns None when there is no HloModule header."""
    lines = text.splitlines()
    i, preamble = 0, []
    header = None
    while i < len(lines):
        hm = _HEADER_RE.match(lines[i])
        if hm:
            header = hm.group(1)
            i += 1
            break
        preamble.append(lines[i])
        i += 1
    if header is None:
        # No header at all: the parser accepts that too (a module named by
        # default, e.g. `e { ... }` in the CPU benchmark inputs — 319 of
        # 517 .hlo files). Start over, keeping comment lines as preamble.
        header, i, preamble = "", 0, []
        while i < len(lines) and not _COMP_OPEN_RE.match(lines[i]):
            preamble.append(lines[i])
            i += 1
        if i >= len(lines):
            return None
        module = HloModule(name="module", header_attrs="", preamble=preamble)
        module.had_header = False
    else:
        module = None
    # A header may continue on following lines until a blank line or the
    # first computation.
    while module is None and i < len(lines) and lines[i].strip() \
            and not _COMP_OPEN_RE.match(lines[i]) \
            and not lines[i].strip().startswith("//"):
        header += " " + lines[i].strip()
        i += 1
    if module is None:
        hm = re.match(r'\s*([\w.\-]+)?(.*)$', header, re.S)
        mod_name = (hm.group(1) or "module") if hm else "module"
        attrs = (hm.group(2) or "").rstrip() if hm else ""
        module = HloModule(name=mod_name, header_attrs=attrs, preamble=preamble)

    leading: List[str] = []
    cur: Optional[HloComputation] = None
    pending: Optional[str] = None

    def flush_pending():
        nonlocal pending
        if cur is not None and pending is not None:
            ins = parse_instruction(pending)
            if ins is not None:
                cur.instrs.append(ins)
        pending = None

    while i < len(lines):
        line = lines[i]
        i += 1
        if cur is None:
            om = _COMP_OPEN_RE.match(line)
            if om:
                cur = HloComputation(name=om.group(3), is_entry=bool(om.group(1)),
                                     signature=(om.group(4).strip() if om.group(4) else None),
                                     leading=[l for l in leading if l.strip()])
                leading = []
            else:
                leading.append(line)
            continue
        if _COMP_CLOSE_RE.match(line) and not (pending is not None and pending.count("{") > pending.count("}")):
            flush_pending()
            module.computations.append(cur)
            cur = None
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if pending is not None and pending.count("{") > pending.count("}"):
            # An instruction with an inline computation body (`calls={
            # ... }`, `to_apply={ ... }`) spans lines until its braces
            # balance; the lines inside are that computation's, not the
            # enclosing one's.
            pending += " " + stripped
        elif _INSTR_START_RE.match(stripped):
            flush_pending()
            pending = stripped
        elif pending is not None:
            pending += " " + stripped
    if cur is not None:            # unterminated computation: keep what parsed
        flush_pending()
        module.computations.append(cur)
    return module


# ---------------------------------------------------------------------------
# Rewriting helpers used by the fusion strategies
# ---------------------------------------------------------------------------

def rename_operands(instr: HloInstr, mapping: dict) -> HloInstr:
    """A copy of `instr` with operand names (and control-predecessor names)
    rewritten through `mapping`; its own name too if mapped."""
    ops = []
    for tok in instr.operands:
        n = operand_name(tok)
        if n and n in mapping:
            new = mapping[n]
            m = _NAME_RE.search(tok.strip())
            ops.append(tok.strip()[:m.start()] + "%" + new)
        else:
            ops.append(tok)
    attrs = instr.attrs
    if "control-predecessors" in attrs:
        def _sub(m):
            names = [f"%{mapping.get(n, n)}" for n in re.findall(r'%?([\w.\-]+)', m.group(1))]
            return "control-predecessors={" + ", ".join(names) + "}"
        attrs = _CTRL_PRED_RE.sub(_sub, attrs)
    return HloInstr(name=mapping.get(instr.name, instr.name), shape=instr.shape,
                    opcode=instr.opcode, operands=ops, attrs=attrs,
                    is_root=instr.is_root, comment=instr.comment)


def rename_computation_refs(instr: HloInstr, mapping: dict) -> HloInstr:
    """A copy with computation references in attributes renamed."""
    def _sub(m):
        key, val = m.group(1), m.group(2)
        if val.startswith("{"):
            names = re.findall(r'%?([\w.\-]+)', val[1:-1])
            return f"{key}={{" + ", ".join(f"%{mapping.get(n, n)}" for n in names) + "}"
        n = val.lstrip("%")
        return f"{key}=%{mapping.get(n, n)}"
    attrs = _COMP_REF_RE.sub(_sub, instr.attrs)
    return HloInstr(name=instr.name, shape=instr.shape, opcode=instr.opcode,
                    operands=list(instr.operands), attrs=attrs,
                    is_root=instr.is_root, comment=instr.comment)


def set_param_index(instr: HloInstr, index: int) -> HloInstr:
    return HloInstr(name=instr.name, shape=instr.shape, opcode=instr.opcode,
                    operands=[str(index)], attrs=instr.attrs,
                    is_root=instr.is_root, comment=instr.comment)


def strip_header_attr(attrs: str, key: str) -> str:
    """Remove `, key=value` (value possibly brace-nested) from a header."""
    m = re.search(r',\s*' + re.escape(key) + r'\s*=\s*', attrs)
    if not m:
        return attrs
    start = m.start()
    j = m.end()
    if j < len(attrs) and attrs[j] == "{":
        end = _balanced_end(attrs, j, "{", "}")
        j = end + 1 if end != -1 else len(attrs)
    else:
        while j < len(attrs) and attrs[j] not in ",":
            j += 1
    return attrs[:start] + attrs[j:]


def header_attr(attrs: str, key: str) -> Optional[str]:
    m = re.search(r'(?:^|,)\s*' + re.escape(key) + r'\s*=\s*([^,{]+|\{)', attrs)
    return m.group(1).strip() if m else None


def unique_name(base: str, taken: set) -> str:
    if base not in taken:
        return base
    k = 1
    while f"{base}.{k}" in taken:
        k += 1
    return f"{base}.{k}"
