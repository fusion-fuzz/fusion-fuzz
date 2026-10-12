"""
core/lean_text.py — reading and writing Lean 4 source, for the Lean
adapter's parser (projects/lean/parser.py), its three fusion strategies
(core/fusion.py) and state analysis (core/state_analysis.py), so all of
them cut a file the same way.

A Lean file is a header (`module`, `prelude`, `import ...`) followed by a
flat sequence of *commands*. By convention every command starts in column
zero and everything that continues it (match alternatives `| ...`,
`where`, `deriving`, `termination_by`, structure fields, the body of a
`by` or `do` block) is either indented or starts with a continuation
token, so a command boundary can be found line by line once comments and
strings are masked out. Doc comments (`/-- -/`), attributes (`@[...]`),
and the `... in` prefixes (`#guard_msgs in`, `set_option x v in`,
`open X in`) belong to the command that follows them.

Scopes (`namespace`/`section`/`mutual` ... `end`) are tracked so that a
cut can close the scopes open at that point and reopen them afterwards,
replaying the `open`/`variable`/`universe`/`set_option` commands that were
in effect — without that, interleaving two programs in the middle of a
namespace produced "invalid 'end', name mismatch".
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lexical mask
# ---------------------------------------------------------------------------

_IDENT_START = r"[^\W\dλΠΣ]"     # a letter or `_`, not λ Π Σ
_IDENT_REST = r"[\w'!?⁰-₟]"
NAME_PAT = rf"{_IDENT_START}{_IDENT_REST}*(?:\.{_IDENT_START}{_IDENT_REST}*)*"
IDENT_RE = re.compile(rf"(?<![\w'!?.«])(?:{NAME_PAT})")


@dataclass
class Comment:
    start: int          # offset of `--` / `/-`
    end: int            # offset just past the comment
    doc: bool           # `/--` (decl doc) or `/-!` (module doc)


def scan(content: str) -> Tuple[str, List[Comment]]:
    """(masked, comments): `content` with the interior of every comment,
    string and char literal replaced by spaces (newlines kept, so line and
    column numbers are unchanged), plus the comment spans.

    Block comments nest in Lean (`/- /- -/ -/`). `'` is a char literal only
    where an identifier cannot be continuing (`x'`, `h'` are names), and a
    raw string `r#"..."#` ends at the matching number of `#`s.
    """
    out = list(content)
    comments: List[Comment] = []
    n = len(content)
    i = 0

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = content[i]
        if c == "-" and content.startswith("--", i):
            j = content.find("\n", i)
            j = n if j == -1 else j
            comments.append(Comment(i, j, False))
            blank(i, j)
            i = j
        elif c == "/" and content.startswith("/-", i):
            doc = content.startswith("/--", i) or content.startswith("/-!", i)
            depth, j = 1, i + 2
            while j < n and depth:
                if content.startswith("/-", j):
                    depth += 1
                    j += 2
                elif content.startswith("-/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            comments.append(Comment(i, j, doc))
            blank(i, j)
            i = j
        elif c == "r" and re.match(r'r#*"', content[i:i + 12]) and \
                (i == 0 or not re.match(r"[\w'!?.]", content[i - 1])):
            m = re.match(r'r(#*)"', content[i:])
            close = '"' + m.group(1)
            j = content.find(close, i + m.end())
            j = n if j == -1 else j + len(close)
            blank(i + 1, j)
            i = j
        elif c == '"':
            j = i + 1
            while j < n:
                if content[j] == "\\":
                    j += 2
                    continue
                if content[j] == '"':
                    j += 1
                    break
                j += 1
            blank(i, j)
            i = j
        elif c == "'" and (i == 0 or not re.match(r"[\w'!?]", content[i - 1])):
            m = re.match(r"'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^\\'\n])'", content[i:i + 12])
            if m:
                blank(i, i + m.end())
                i += m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out), comments


def mask(content: str) -> str:
    return scan(content)[0]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

DECL_KEYWORDS = ("def", "theorem", "lemma", "abbrev", "instance", "example",
                 "axiom", "opaque", "structure", "class", "inductive",
                 "coinductive")
MODIFIERS = ("private", "protected", "public", "noncomputable", "partial",
             "unsafe", "nonrec", "scoped", "local", "meta")
SCOPE_OPENERS = ("namespace", "section", "mutual")

COMMAND_KEYWORDS = DECL_KEYWORDS + MODIFIERS + (
    "mutual", "namespace", "section", "end", "open", "export", "variable",
    "universe", "universes", "set_option", "attribute", "macro",
    "macro_rules", "syntax", "elab", "elab_rules", "notation", "infix",
    "infixl", "infixr", "prefix", "postfix", "declare_syntax_cat",
    "initialize", "builtin_initialize", "run_cmd", "run_meta", "run_elab",
    "omit", "include", "add_decl_doc", "register_simp_attr",
    "register_option", "register_builtin_option", "simproc", "dsimproc",
    "builtin_simproc", "builtin_dsimproc", "grind_pattern",
    "binder_predicate", "unif_hint", "import", "module", "prelude",
    "recommended_spelling", "deriving", "builtin_initialize", "declare_config_elab",
    "irreducible_def", "add_aesop_rules", "seal", "unseal", "show_panel_widgets",
    "builtin_simproc_decl", "simproc_decl", "theorem", "inductive",
)

# `deriving Repr` in column zero continues an inductive; only
# `deriving instance ... for ...` is a command of its own.
_CMD_START_RE = re.compile(
    r"^(?:@\[|#[a-z_]+\b|(?:%s)\b)" % "|".join(sorted(set(COMMAND_KEYWORDS) - {"deriving"},
                                                       key=len, reverse=True))
    + r"|^deriving\s+instance\b")
_PREFIX_ONLY_RE = re.compile(
    r"^(?:@\[[^\]]*\](?:\s*@\[[^\]]*\])*"
    r"|#guard_msgs\b(?:\s*\([^)]*\))?\s+in"
    r"|(?:set_option\s+\S+\s+\S+|open\s+[^\n]+?|omit\s+[^\n]+?|include\s+[^\n]+?)\s+in"
    r"|(?:(?:%s)\s*)+)\s*$" % "|".join(MODIFIERS))


@dataclass
class Command:
    start: int                    # first line (index) of the command, prefixes included
    end: int                      # exclusive end line
    keyword: str                  # first keyword after prefixes/modifiers
    head_line: int                # line holding the keyword
    text: str = ""
    name: Optional[str] = None    # declared name, as written
    fqname: Optional[str] = None  # with the enclosing namespaces
    namespace: str = ""           # namespace the command sits in
    modifiers: List[str] = field(default_factory=list)
    depth: int = 0                # number of open scopes before it


_IN_PREFIX_RE = re.compile(
    r"(?:#guard_msgs\b(?:\s*\([^)]*\))?|set_option\s+\S+\s+\S+|open\s+[^\n]*?|"
    r"omit\s+[^\n]*?|include\s+[^\n]*?)\s+in\b")


def _skip_prefixes(code: str) -> Tuple[int, List[str]]:
    """(offset of the command keyword in masked `code`, modifiers seen):
    skips whitespace (doc comments are already blank), `@[...]` attribute
    lists (brackets nest), modifiers and the `... in` prefixes."""
    i, n, mods = 0, len(code), []
    while i < n:
        while i < n and code[i].isspace():
            i += 1
        if code.startswith("@[", i):
            depth, j = 0, i + 1
            while j < n:
                if code[j] == "[":
                    depth += 1
                elif code[j] == "]":
                    depth -= 1
                    if depth == 0:
                        j += 1
                        break
                j += 1
            i = j
            continue
        m = re.match(r"(%s)\b" % "|".join(MODIFIERS), code[i:])
        if m:
            mods.append(m.group(1))
            i += m.end()
            continue
        m = _IN_PREFIX_RE.match(code, i)
        if m:
            i = m.end()
            continue
        break
    return i, mods


def _line_offsets(content: str) -> List[int]:
    offs, off = [], 0
    for ln in content.split("\n"):
        offs.append(off)
        off += len(ln) + 1
    return offs


def command_start_lines(content: str, masked: Optional[str] = None) -> List[int]:
    """Line indices that begin a command (before prefix merging)."""
    if masked is None:
        masked, comments = scan(content)
    else:
        comments = scan(content)[1]
    lines = content.split("\n")
    mlines = masked.split("\n")
    offs = _line_offsets(content)
    doc_lines = set()
    for c in comments:
        if c.doc and content.startswith("/--", c.start):
            ln = content.count("\n", 0, c.start)
            if offs[ln] == c.start:      # a decl doc comment opening in column zero
                doc_lines.add(ln)
    starts = []
    for i, (raw, ml) in enumerate(zip(lines, mlines)):
        if i in doc_lines or (ml[:1].strip() and _CMD_START_RE.match(ml)):
            starts.append(i)
        elif raw.startswith("/-!") :
            starts.append(i)
    return starts


def _is_prefix_only(raw_lines: List[str], mlines: List[str], a: int, b: int) -> bool:
    """Is the command candidate spanning lines [a, b) nothing but a prefix
    of the next one — a doc comment, attributes, modifiers, `... in`?"""
    code = "\n".join(mlines[a:b])
    if not code.strip():
        return raw_lines[a].lstrip().startswith("/--")
    off, _mods = _skip_prefixes(code)
    return not code[off:].strip()


def split_commands(content: str) -> List[Command]:
    """Commands of `content` in order, with doc comments, attributes and
    `... in` prefixes merged into the command they belong to. Lines before
    the first command (the header and leading comments) are not part of
    any command."""
    masked, comments = scan(content)
    lines = content.split("\n")
    mlines = masked.split("\n")
    offs = _line_offsets(content)
    starts = command_start_lines(content, masked)
    if not starts:
        return []
    bounds = list(zip(starts, starts[1:] + [len(lines)]))
    merged: List[Tuple[int, int]] = []
    pending: Optional[int] = None
    for a, b in bounds:
        if _is_prefix_only(lines, mlines, a, b):
            if pending is None:
                pending = a
            continue
        merged.append((pending if pending is not None else a, b))
        pending = None
    if pending is not None:
        merged.append((pending, len(lines)))

    cmds: List[Command] = []
    for a, b in merged:
        # trailing blank lines stay with the command (keeps text contiguous)
        head = a
        while head < b and not mlines[head].strip():
            head += 1
        code = "\n".join(mlines[head:b])
        kw_off, mods = _skip_prefixes(code)
        mk = re.match(r"#?[A-Za-z_]\w*(?:\s+instance\b)?", code[kw_off:])
        kw = mk.group(0) if mk else ""
        if not kw and lines[a].lstrip().startswith("/-!"):
            kw = "moddoc"
        # The keyword's own line (after a multi-line attribute list).
        hl = head + code.count("\n", 0, kw_off)
        cmd = Command(start=a, end=b, keyword=kw.split()[0] if kw else "", head_line=hl,
                      text="\n".join(lines[a:b]), modifiers=mods)
        if kw.startswith("deriving") and kw.endswith("instance"):
            cmd.keyword = "deriving"
        cmds.append(cmd)
    _annotate_scopes(cmds, mlines)
    return cmds


_NAME_AFTER_KW_RE = re.compile(
    r"(?:def|theorem|lemma|abbrev|instance|axiom|opaque|structure|class(?:\s+inductive)?|"
    r"inductive|coinductive)\s+(?:\([^)]*\)\s*)?(?:\[[^\]]*\]\s*)?"
    r"(?P<name>(?:_root_\.)?" + NAME_PAT + r")")


def _decl_name(cmd: Command, mlines: List[str]) -> Optional[str]:
    if cmd.keyword not in DECL_KEYWORDS or cmd.keyword == "example":
        return None
    seg = "\n".join(mlines[cmd.head_line:cmd.end])
    m = _NAME_AFTER_KW_RE.search(seg)
    if not m:
        return None
    name = m.group("name")
    if cmd.keyword == "instance":
        # `instance : Foo Bar` has no name; `instance foo : ...` has one.
        after = seg[m.start("name") - 1:m.start("name")]
        rest = seg[m.end("name"):m.end("name") + 3]
        if name in ("priority",) or rest.lstrip().startswith(")"):
            return None
    return name


def _annotate_scopes(cmds: List[Command], mlines: List[str]) -> None:
    """Fill in namespace, depth, name and fqname for every command."""
    stack: List[Tuple[str, str]] = []           # (kind, name)
    for cmd in cmds:
        cmd.depth = len(stack)
        cmd.namespace = ".".join(n for k, n in stack if k == "namespace" and n)
        head = mlines[cmd.head_line].strip() if cmd.head_line < len(mlines) else ""
        if cmd.keyword in ("namespace", "section"):
            # `public section`, `noncomputable section`: the name follows
            # the keyword, wherever it is on the line.
            toks = head.split()
            k = toks.index(cmd.keyword) if cmd.keyword in toks else len(toks)
            name = toks[k + 1] if k + 1 < len(toks) else ""
            if cmd.keyword == "namespace":
                for comp in name.split("."):
                    stack.append(("namespace", comp))
            else:
                stack.append(("section", name))
        elif cmd.keyword == "mutual":
            stack.append(("mutual", ""))
        elif cmd.keyword == "end":
            toks = head.split()
            name = toks[1] if len(toks) > 1 else ""
            ncomp = len(name.split(".")) if name else 1
            for _ in range(min(ncomp, len(stack))):
                stack.pop()
        name = _decl_name(cmd, mlines)
        if name:
            cmd.name = name
            if name.startswith("_root_."):
                cmd.fqname = name[len("_root_."):]
            else:
                cmd.fqname = f"{cmd.namespace}.{name}" if cmd.namespace else name


def scope_stack_before(cmds: List[Command], index: int, content_lines: List[str]) -> List[Dict]:
    """The scopes open just before command `index`, outermost first, each
    with the context commands (`open`, `variable`, `universe`,
    `set_option`, `attribute [local ...]`) issued directly inside it —
    what has to be replayed to reopen it."""
    stack: List[Dict] = [{"kind": "root", "name": "", "context": []}]
    for cmd in cmds[:index]:
        head = content_lines[cmd.head_line].strip() if cmd.head_line < len(content_lines) else ""
        toks = head.split()
        if cmd.keyword in ("namespace", "section") and cmd.keyword in toks:
            k = toks.index(cmd.keyword)
            name = toks[k + 1] if k + 1 < len(toks) else ""
            stack.append({"kind": cmd.keyword, "name": name, "header": head, "context": []})
        elif cmd.keyword == "mutual":
            stack.append({"kind": "mutual", "name": "", "header": "mutual", "context": []})
        elif cmd.keyword == "end":
            name = toks[1] if len(toks) > 1 else ""
            if len(stack) > 1:
                top = stack[-1]
                if top["kind"] == "namespace" and name and top["name"] != name and \
                        "." in name:
                    # `end A.B` closes `namespace A.B` opened as one
                    pass
                stack.pop()
        elif cmd.keyword in ("open", "variable", "universe", "universes", "set_option",
                             "attribute", "omit", "include") and \
                not re.search(r"\bin\s*$", head):
            stack[-1]["context"].append(cmd.text.rstrip())
    return stack[1:]


def close_scopes(stack: List[Dict]) -> List[str]:
    return [("end" if s["kind"] == "mutual" else f"end {s['name']}".rstrip())
            for s in reversed(stack)]


def reopen_scopes(stack: List[Dict]) -> List[str]:
    out = []
    for s in stack:
        if s["kind"] == "mutual":
            out.append("mutual")
        else:
            out.append(s.get("header") or f"{s['kind']} {s['name']}".rstrip())
        out.extend(s["context"])
    return out


def open_scopes_at_end(content: str) -> List[Dict]:
    cmds = split_commands(content)
    return scope_stack_before(cmds, len(cmds), content.split("\n"))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

_HEADER_LINE_RE = re.compile(
    r"^\s*(?:module\b|prelude\b|(?:public\s+)?(?:meta\s+)?import\b)")


def split_header(content: str) -> Tuple[List[str], str]:
    """(header_lines, body). The header is `module`/`prelude`/`import`
    lines before the first command; comments and blank lines between them
    stay in the body (in front of it)."""
    masked = mask(content)
    lines = content.split("\n")
    mlines = masked.split("\n")
    header, keep = [], []
    in_header = True
    for raw, ml in zip(lines, mlines):
        if in_header and _HEADER_LINE_RE.match(ml) and ml.strip():
            header.append(raw.rstrip())
            continue
        if in_header and ml.strip() and not _HEADER_LINE_RE.match(ml):
            in_header = False
        keep.append(raw)
    return header, "\n".join(keep).strip("\n")


def merge_headers(*headers: List[str]) -> List[str]:
    """One header for a fused file: `module` first when any parent is a
    module (its `public`/`meta` imports and `@[expose]` need it), then
    every import once. Without `module`, the module-system import forms
    are rewritten to plain `import`."""
    is_module = any(h.strip() == "module" for hs in headers for h in hs)
    imports = []
    for hs in headers:
        for h in hs:
            s = h.strip()
            if s in ("module", "prelude") or not s:
                continue
            if not is_module:
                s = re.sub(r"^(?:public\s+)?(?:meta\s+)?import\s+(?:all\s+)?", "import ", s)
            imports.append(s)
    out = ["module"] if is_module else []
    out.extend(dict.fromkeys(imports))
    return out


def assemble(header: List[str], *bodies: str) -> str:
    parts = []
    if header:
        parts.append("\n".join(header))
    parts.extend(b.strip("\n") for b in bodies if b and b.strip())
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass
class Decl:
    kind: str                 # def/theorem/abbrev/instance/structure/...
    name: str                 # as written
    fqname: str               # namespace-qualified
    binders: str              # text between the name and the type colon
    type: str                 # declared type (after the top-level `:`), "" if none
    explicit_arity: int       # number of explicit `(x : T)` binders
    private: bool
    namespace: str
    command_index: int
    depth: int
    params_free: bool         # a type declaration with no explicit parameters


_BINDER_RE = re.compile(r"\(([^():]+):([^()]*(?:\([^()]*\)[^()]*)*)\)")


def _top_level_colon_split(header: str) -> Tuple[str, str]:
    """Split a declaration header (text after the name, before `:=`/`where`/
    `|`) into (binders, type) at the first `:` at bracket depth zero."""
    depth = 0
    for i, ch in enumerate(header):
        if ch in "([{⟨⦃":
            depth += 1
        elif ch in ")]}⟩⦄":
            depth -= 1
        elif ch == ":" and depth == 0 and not header.startswith(":=", i):
            return header[:i], header[i + 1:]
    return header, ""


def _decl_header_text(seg: str, name_end: int) -> str:
    """Text after the declared name up to the body: `:=`, ` where`, a
    column-zero `|`, or `extends`/`deriving` for type declarations."""
    rest = seg[name_end:]
    stops = [rest.find(":=")] + [m.start() for m in re.finditer(
        r"\bwhere\b|\n\||\n\s*\|\s|\bextends\b|\bderiving\b|:=", rest)]
    stops = [s for s in stops if s >= 0]
    end = min(stops) if stops else len(rest)
    return rest[:end]


def declarations(content: str, cmds: Optional[List[Command]] = None) -> List[Decl]:
    masked = mask(content)
    mlines = masked.split("\n")
    if cmds is None:
        cmds = split_commands(content)
    out = []
    for idx, cmd in enumerate(cmds):
        if not cmd.name:
            continue
        seg = "\n".join(mlines[cmd.head_line:cmd.end])
        m = _NAME_AFTER_KW_RE.search(seg)
        if not m:
            continue
        header = _decl_header_text(seg, m.end("name"))
        binders, typ = _top_level_colon_split(header)
        explicit = len(_BINDER_RE.findall(binders))
        out.append(Decl(
            kind=cmd.keyword, name=cmd.name, fqname=cmd.fqname or cmd.name,
            binders=" ".join(binders.split()), type=" ".join(typ.split()),
            explicit_arity=explicit, private="private" in cmd.modifiers,
            namespace=cmd.namespace, command_index=idx, depth=cmd.depth,
            params_free=(explicit == 0 and not re.search(r"[({\[⦃]", binders))))
    return out


TYPE_KINDS = ("structure", "class", "inductive", "coinductive")
PROOF_KINDS = ("theorem", "lemma")
TERM_KINDS = ("def", "abbrev", "opaque", "axiom", "instance")


def kind_class(kind: str) -> str:
    if kind in TYPE_KINDS:
        return "type"
    if kind in PROOF_KINDS:
        return "proof"
    return "term"


def constructors(cmd_text: str) -> List[str]:
    """Constructor names of an inductive command (`| name ...` lines)."""
    return re.findall(r"^\s*\|\s*(" + NAME_PAT + r")",
                      mask(cmd_text), re.M)


def structure_fields(cmd_text: str) -> List[str]:
    """Field names of a structure command (indented `name : T` lines after
    `where`, or parenthesised `(name : T)` binders in the old syntax)."""
    m = mask(cmd_text)
    body = m.split("where", 1)[1] if re.search(r"\bwhere\b", m) else ""
    names = re.findall(r"^\s+([A-Za-z_][\w']*(?:\s+[A-Za-z_][\w']*)*)\s*:", body, re.M)
    out = []
    for n in names:
        out.extend(n.split())
    return out


# ---------------------------------------------------------------------------
# Identifiers
# ---------------------------------------------------------------------------

KEYWORDS = frozenset("""
def theorem lemma abbrev instance example axiom opaque structure class inductive
where extends deriving namespace section end open export variable universe
fun let have show from by do then else if match with at in return for unless
mut try catch finally calc suffices obtain mutual private protected public
noncomputable partial unsafe macro syntax elab notation set_option attribute
Type Sort Prop true false this sorry admit import module prelude termination_by
decreasing_by nomatch nofun intro intros exact apply rfl simp simp_all omega decide
rw rewrite cases rcases induction constructor refine assumption contradiction
exfalso trivial unfold refine' exists use left right next case all_goals
any_goals first repeat done skip congr ext funext subst injection grind
native_decide simpa by_cases specialize generalize revert clear rename_i change
dsimp conv rwa exact? apply? aesop norm_num linarith ring positivity
some none pure bind
""".split())


def identifiers(code: str) -> List[str]:
    """Bare identifiers of `code` (comments and strings masked), in order of
    first appearance, keywords excluded."""
    m = mask(code)
    out = []
    for tok in IDENT_RE.findall(m):
        if tok in KEYWORDS:
            continue
        out.append(tok)
    return list(dict.fromkeys(out))


def ident_pattern(name: str) -> re.Pattern:
    return re.compile(rf"(?<![\w'!?.«]){re.escape(name)}(?![\w'!?])")


def replace_ident(code: str, old: str, new: str, positions=None) -> str:
    """Replace every whole-identifier occurrence of `old` outside comments
    and strings (or only those at `positions`, a set of start offsets)."""
    m = mask(code)
    pat = ident_pattern(old)
    hits = [x for x in pat.finditer(m) if positions is None or x.start() in positions]
    for x in reversed(hits):
        code = code[:x.start()] + new + code[x.end():]
    return code


def rename_collisions(donor: str, host: str, suffix: str = "_ffl") -> Tuple[str, Dict[str, str]]:
    """Rename every declaration of `donor` whose name would collide with
    one of `host`'s (same fully-qualified name, or same short name — a
    short name is enough to make an unqualified reference ambiguous once
    both namespaces are open). Returns (renamed_donor, mapping)."""
    host_cmds = split_commands(host)
    host_names = set()
    for c in host_cmds:
        if c.fqname:
            host_names.add(c.fqname)
            host_names.add(c.fqname.split(".")[-1])
        if c.keyword in ("declare_syntax_cat",):
            toks = mask(c.text).split()
            if len(toks) > 1:
                host_names.add(toks[1])
    mapping: Dict[str, str] = {}
    for c in split_commands(donor):
        short = (c.name or "").split(".")[-1]
        if not short:
            continue
        if (c.fqname in host_names or short in host_names) and short not in mapping:
            k = 1
            new = f"{short}{suffix}"
            while new in host_names or new in mapping.values():
                k += 1
                new = f"{short}{suffix}{k}"
            mapping[short] = new
    out = donor
    for old, new in mapping.items():
        out = _replace_name_component(out, old, new)
    return out, mapping


def _replace_name_component(code: str, old: str, new: str) -> str:
    """Rename `old` wherever it is a whole identifier *or* the last
    component of a dotted one (`Foo.old`, the declared name `Ns.old`), and
    where it is the first component (`old.mk`, `old.rec`): renaming a type
    must carry its constructors' qualified references along."""
    m = mask(code)
    pat = re.compile(rf"(?<![\w'!?«]){re.escape(old)}(?![\w'!?])")
    hits = [x for x in pat.finditer(m)]
    for x in reversed(hits):
        code = code[:x.start()] + new + code[x.end():]
    return code


# ---------------------------------------------------------------------------
# Blocks (`by` / `do`)
# ---------------------------------------------------------------------------

@dataclass
class Block:
    kind: str           # "by" or "do"
    opener_line: int    # line ending in `by` / `do`
    start: int          # first body line
    end: int            # exclusive
    indent: int         # body base indentation
    stmt_lines: List[int] = field(default_factory=list)   # lines starting a statement


_BLOCK_OPEN_RE = re.compile(r"(?:^|[\s(:=])(?P<kw>by|do)\s*$")
# Clauses of the enclosing declaration, not statements of the block.
_BLOCK_END_RE = re.compile(r"(?:termination_by|decreasing_by|where|deriving)\b")
_CONT_START_RE = re.compile(r"^(?:\||<;>|\)|\]|\}|⟩|\.\.|·\s*$|with\b|then\b|else\b|<\|>|_\s*=|_\s*≤|_\s*<|_\s*↔)")


def blocks(content: str) -> List[Block]:
    """Multi-line `by`/`do` blocks: an opener at the end of a line followed
    by lines indented deeper than that line. A statement starts at a body
    line with exactly the base indentation that does not continue the
    previous one (`| alt`, `<;>`, `then`, `else`)."""
    masked = mask(content)
    lines = content.split("\n")
    mlines = masked.split("\n")
    out = []
    for i, ml in enumerate(mlines):
        m = _BLOCK_OPEN_RE.search(ml.rstrip())
        if not m:
            continue
        open_indent = len(ml) - len(ml.lstrip())
        j = i + 1
        while j < len(lines) and not mlines[j].strip():
            j += 1
        if j >= len(lines):
            continue
        base = len(mlines[j]) - len(mlines[j].lstrip())
        if base <= open_indent and not (open_indent == 0 and base > 0):
            continue
        if base == 0:
            continue
        end = j
        stmts = []
        k = j
        while k < len(lines):
            mk = mlines[k]
            if mk.strip():
                ind = len(mk) - len(mk.lstrip())
                if ind < base or _BLOCK_END_RE.match(mk.lstrip()):
                    break
                if ind == base and not _CONT_START_RE.match(mk.lstrip()):
                    stmts.append(k)
            end = k + 1
            k += 1
        if stmts:
            out.append(Block(m.group("kw"), i, j, end, base, stmts))
    return out


def reindent(lines: List[str], old: int, new: int) -> List[str]:
    out = []
    for ln in lines:
        if not ln.strip():
            out.append("")
            continue
        cur = len(ln) - len(ln.lstrip())
        out.append(" " * max(0, cur - old + new) + ln.lstrip())
    return out


_HYP_INTRO_RE = re.compile(
    r"\b(?:intro|intros|rintro|obtain|rcases|cases|induction|have|let|set|"
    r"generalize|next|case|fun)\b([^\n:=]*)")


def hypotheses(lines: List[str]) -> List[str]:
    """Names a tactic/do prefix introduces (a cheap over-approximation)."""
    out = []
    for ln in lines:
        m = mask(ln)
        for g in _HYP_INTRO_RE.findall(m):
            for tok in IDENT_RE.findall(g):
                if tok not in KEYWORDS and (tok[0].islower() or tok[0] == "_"):
                    out.append(tok)
        for g in re.findall(r"\blet\s+(?:mut\s+)?([A-Za-z_][\w']*)", m):
            out.append(g)
        for g in re.findall(r"^\s*([A-Za-z_][\w']*)\s*←", m):
            out.append(g)
    return list(dict.fromkeys(n for n in out if n not in KEYWORDS))


def binder_names(binders: str) -> List[str]:
    out = []
    for grp, _typ in _BINDER_RE.findall(binders):
        out.extend(t for t in grp.split() if re.match(r"^[A-Za-z_][\w']*$", t))
    for grp in re.findall(r"[\[{⦃]([^:\]}⦄]+):", binders):
        out.extend(t for t in grp.split() if re.match(r"^[A-Za-z_][\w']*$", t))
    return list(dict.fromkeys(out))


# ---------------------------------------------------------------------------
# Segment boundaries (state fusion)
# ---------------------------------------------------------------------------

def segment_boundaries(content: str) -> List[int]:
    """Line indices after which `content` can be cut between two commands:
    the next non-blank line starts a command and the cut is not inside a
    `mutual` block. Namespaces and sections may be open at the cut — the
    state strategy closes and reopens them (see scope_stack_before)."""
    cmds = split_commands(content)
    if not cmds:
        return []
    out = []
    for k, cmd in enumerate(cmds[1:], start=1):
        prev = cmds[k - 1]
        if prev.keyword == "mutual":
            continue
        stack = scope_stack_before(cmds, k, content.split("\n"))
        if any(s["kind"] == "mutual" for s in stack):
            continue
        out.append(cmd.start - 1)
    out.append(len(content.split("\n")) - 1)
    return sorted(set(b for b in out if b >= 0))
