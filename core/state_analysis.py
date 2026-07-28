"""
core/state_analysis.py — State-of-Interest Profiling & Safe Splice

Implements the "state" half of state fusion: a *state* is the point in a
seed's execution immediately following a given statement. Rather than
fusing only at a seed's final state (what dataflow fusion effectively
does — it links one whole program to another via a single bridge
variable), state fusion selects an *intermediate* program point from each
seed and grafts one seed's continuation into the other's state there —
richer search space, deeper interactions.

Two things make a point usable:

1. It must be a *state of interest* — likely to interact meaningfully
   with another seed's continuation. Unified rule across all supported
   languages except Haskell: the point(s) with the most *live*
   variables — declared, and still in scope — is the "most complex" state,
   offering a donor continuation the richest possible surface of names to
   collide with. Computed per-language via a small (declare/decrement/
   reset-or-brace-scope) regex config below — a static best-effort
   approximation, not a real dataflow/liveness analysis. Haskell instead
   just picks a uniformly random line (HaskellStateFusionStrategy in
   core/fusion.py) — its layout-based scoping doesn't fit this model, and
   simplicity won out over trying to make it fit.
2. It must be *safe* to splice at — inserting statements there can't
   break the enclosing syntactic/structural integrity (an unterminated
   string, a line mid-expression, wrong indentation for Python). This
   module checks structural safety; it does not (and does not try to)
   guarantee the grafted program is semantically valid — a donor
   continuation referencing the donor's own earlier variables is exactly
   the kind of "reach into an unfamiliar state" case state fusion is
   for, and it is expected to sometimes fail. core/orchestrator.py's
   FuseValidRate tracks how often that happens per language.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# seed metadata "type"/language strings that map onto LIVE_VAR_CONFIGS keys.
LANGUAGE_ALIASES: Dict[str, str] = {
    "python": "cpython", "py": "cpython", "cpython": "cpython",
    "rust": "rust", "rs": "rust",
    "php": "php", "phpt": "php",
    "go": "go",
    "c": "clang", "cpp": "clang", "cxx": "clang", "clang": "clang",
    "swift": "swift",
    "haskell": "haskell", "hs": "haskell", "ghc": "haskell",
    "flang": "flang", "fortran": "flang", "f90": "flang",
    "mlir": "mlir",
}


# ---------------------------------------------------------------------------
# Per-language live-variable-count config
# ---------------------------------------------------------------------------
#
# Two scoping modes, chosen per language by how it actually scopes names:
#
# "flat"  — the language doesn't scope variables to ordinary control-flow
#           blocks (if/for/while/DO), only to function/subroutine
#           boundaries (cpython, php) or not at all within one program
#           unit (flang) — so a single running set, reset at each new
#           top-level unit, is already an accurate model, not a
#           simplification.
# "brace" — the language closes a variable's scope at `}` (rust, clang,
#           swift) or, for mlir, an SSA value's scope ends at its
#           enclosing region's `}` — modeled as a stack of scope frames,
#           pushed on `{` and popped (subtracting its names) on `}`.
#
# `declare`/`decrement` regexes each have exactly one capturing group: the
# variable name introduced/removed. `reset` (flat mode only) marks a line
# that starts a new top-level unit, clearing the live set. `multi_name`
# splits the declare group on commas (Fortran's `TYPE :: a, b, c`).
# `case_insensitive` folds names to uppercase before set membership
# (Fortran identifiers are case-insensitive).

@dataclass
class LiveVarConfig:
    mode: str
    declare: List[str]
    decrement: List[str] = field(default_factory=list)
    reset: List[str] = field(default_factory=list)
    multi_name: bool = False
    case_insensitive: bool = False


LIVE_VAR_CONFIGS: Dict[str, LiveVarConfig] = {
    "cpython": LiveVarConfig(
        mode="flat",
        declare=[r'^\s*([A-Za-z_]\w*)\s*=(?!=)',
                 r'\bfor\s+([A-Za-z_]\w*)\s+in\b',
                 r'\bwith\b.*\bas\s+([A-Za-z_]\w*)\s*:'],
        decrement=[r'\bdel\s+([A-Za-z_]\w*)'],
        reset=[r'^(?:def|class)\s+\w+'],
    ),
    "php": LiveVarConfig(
        mode="flat",
        declare=[r'\$([A-Za-z_]\w*)\s*=(?!=)',
                 r'\bas\s+&?\$([A-Za-z_]\w*)\b'],
        decrement=[r'\bunset\s*\(\s*\$([A-Za-z_]\w*)'],
        reset=[r'^\s*function\s+\w+\s*\('],
    ),
    "flang": LiveVarConfig(
        mode="flat",
        declare=[r'(?i)\b(?:integer|real|double\s+precision|complex|logical|character)\b'
                 r'\s*(?:\([^)]*\)|\*\d+)?\s*(?:,[^:]*)?::\s*(.+)$'],
        reset=[r'(?i)^\s*(?:program|subroutine|function)\s+\w+'],
        multi_name=True,
        case_insensitive=True,
    ),
    # Go isn't covered by any FusionStrategy's state fusion today (no
    # GoFusionStrategy exists), so it's intentionally omitted here — add a
    # config if/when Go state fusion is implemented.
    "rust": LiveVarConfig(
        mode="brace",
        declare=[r'\blet\s+(?:mut\s+)?([A-Za-z_]\w*)'],
        # `drop(x)` genuinely ends x's lifetime in Rust (a move) — unlike
        # clang's free()/swift's deinit, this one is safe to model as a
        # real decrement rather than relying on scope-exit alone.
        decrement=[r'\bdrop\s*\(\s*([A-Za-z_]\w*)\s*\)'],
    ),
    "clang": LiveVarConfig(
        mode="brace",
        declare=[r'\b(?:_Bool|bool|char|short(?:\s+int)?|int|long(?:\s+long)?(?:\s+int)?|'
                 r'float|double|long\s+double|unsigned(?:\s+\w+)?|signed(?:\s+\w+)?|'
                 r'size_t|ssize_t|u?int(?:8|16|32|64)_t)\b'
                 r'\s*\*{0,2}\s*([A-Za-z_]\w*)\s*(?:=(?!=)|;|,|\))'],
        # free() invalidates the pointee, not the pointer variable itself —
        # it stays in scope (and in the count) until its enclosing `}`.
    ),
    "swift": LiveVarConfig(
        mode="brace",
        declare=[r'\b(?:var|let)\s+([A-Za-z_]\w*)'],
    ),
    # MLIR is SSA: a %value's scope is exactly its enclosing region, so the
    # brace-stack model is exact here, not a heuristic — no decrement
    # needed since SSA values are never undefined once in scope.
    "mlir": LiveVarConfig(
        mode="brace",
        declare=[r'(%[A-Za-z0-9_$.]+)\s*='],
    ),
}

# ---------------------------------------------------------------------------
# Lexical masking (string/comment aware), shared across languages
# ---------------------------------------------------------------------------

_LEXICON = {
    "rust":    {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"']},
    "php":     {"line": ["//", "#"], "block": [("/*", "*/")], "quotes": ['"', "'"]},
    "cpython": {"line": ["#"], "block": [], "quotes": ['"""', "'''", '"', "'"]},
    "go":      {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"', "`"]},
    "clang":   {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"', "'"]},
    "swift":   {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"']},
    "haskell": {"line": ["--"], "block": [("{-", "-}")], "quotes": ['"']},
    # Fortran uses doubled-quote escaping ('' / "") rather than backslash
    # escaping; the generic backslash-aware scanner below is a reasonable
    # approximation (worst case: it closes a string one quote earlier than
    # a doubled escape would), not exact — fine for a heuristic safety gate.
    "flang":   {"line": ["!"], "block": [], "quotes": ['"', "'"]},
    "mlir":    {"line": ["//"], "block": [], "quotes": ['"']},
}


def _lexical_mask(content: str, language: str) -> List[bool]:
    """bool per character: True where content[i] is real code (outside
    strings/comments). Same purpose as the ad hoc `is_real` masks already
    used by individual fusion strategies (e.g. RustLexMixin), generalized
    across languages via a small per-language token table instead of a
    bespoke scanner per strategy."""
    lex = _LEXICON.get(LANGUAGE_ALIASES.get(language, language), _LEXICON["clang"])
    n = len(content)
    mask = [True] * n
    i = 0
    quotes = sorted(lex["quotes"], key=len, reverse=True)
    while i < n:
        matched = False
        for start_tok, end_tok in lex["block"]:
            if content.startswith(start_tok, i):
                j = content.find(end_tok, i + len(start_tok))
                j = n if j == -1 else j + len(end_tok)
                for k in range(i, j):
                    mask[k] = False
                i = j
                matched = True
                break
        if matched:
            continue
        for tok in lex["line"]:
            if content.startswith(tok, i):
                j = content.find("\n", i)
                j = n if j == -1 else j
                for k in range(i, j):
                    mask[k] = False
                i = j
                matched = True
                break
        if matched:
            continue
        for q in quotes:
            if content.startswith(q, i):
                j = i + len(q)
                while j < n:
                    if content[j] == "\\":
                        j += 2
                        continue
                    if content.startswith(q, j):
                        j += len(q)
                        break
                    j += 1
                else:
                    j = n
                for k in range(i, min(j, n)):
                    mask[k] = False
                i = j
                matched = True
                break
        if matched:
            continue
        i += 1
    return mask


def _paren_depth_at(content: str, mask: List[bool], upto: int) -> int:
    depth = 0
    for i in range(upto):
        if mask[i] and content[i] in "([{":
            depth += 1
        elif mask[i] and content[i] in ")]}":
            depth -= 1
    return depth


# ---------------------------------------------------------------------------
# State point discovery
# ---------------------------------------------------------------------------

@dataclass
class StatePoint:
    line_idx: int            # 0-based line index whose END is the state
    category: str
    matched_text: str
    indent: str = ""

    def to_dict(self) -> dict:
        return {"line_idx": self.line_idx, "category": self.category,
                "matched_text": self.matched_text, "indent": self.indent}

    @staticmethod
    def from_dict(d: dict) -> "StatePoint":
        return StatePoint(d["line_idx"], d["category"], d.get("matched_text", ""), d.get("indent", ""))


# Haskell is deliberately NOT in LIVE_VAR_CONFIGS and has no dedicated
# find_states_of_interest path: its real "many variables" equivalent
# (function-argument/case-alternative/do-bind pattern names, not let/
# where) is abundant enough to count, but its layout-based scoping
# (where/let-in/do blocks scoped by indentation, not braces) makes
# tracking where those go out of scope unreliable enough that it isn't
# worth it — HaskellStateFusionStrategy just always picks a random line
# instead (pick_state_point(..., lightweight=True), unconditionally,
# independent of --pre-analysis).


def _add_name(live: set, name: str, cfg: LiveVarConfig) -> None:
    name = name.strip()
    if not name:
        return
    live.add(name.upper() if cfg.case_insensitive else name)


def _match_pos_ok(mask: List[bool], abs_pos: int) -> bool:
    """A declare/decrement match only counts if its name starts in real
    code (not inside a string/comment)."""
    return abs_pos >= len(mask) or mask[abs_pos]


def _count_live_flat(lines: List[str], line_starts: List[int], mask: List[bool],
                      cfg: LiveVarConfig) -> List[int]:
    """Per-line live-variable count for languages that only scope names to
    a top-level unit (function/subroutine), not to inner control-flow
    blocks — see LiveVarConfig's mode docs."""
    declare_res = [re.compile(p) for p in cfg.declare]
    decrement_res = [re.compile(p) for p in cfg.decrement]
    reset_res = [re.compile(p) for p in cfg.reset]

    live: set = set()
    counts = []
    for idx, line in enumerate(lines):
        start = line_starts[idx]
        if any(r.match(line) for r in reset_res):
            live = set()

        for r in decrement_res:
            for m in r.finditer(line):
                if _match_pos_ok(mask, start + m.start(1)):
                    live.discard(m.group(1).upper() if cfg.case_insensitive else m.group(1))

        for r in declare_res:
            for m in r.finditer(line):
                if not _match_pos_ok(mask, start + m.start(1)):
                    continue
                if cfg.multi_name:
                    for name in m.group(1).split(','):
                        _add_name(live, re.split(r'[\s(]', name.strip(), 1)[0], cfg)
                else:
                    _add_name(live, m.group(1), cfg)

        counts.append(len(live))
    return counts


def _count_live_brace(lines: List[str], line_starts: List[int], mask: List[bool],
                       cfg: LiveVarConfig) -> List[int]:
    """Per-line live-variable count for languages that close a name's
    scope at its enclosing `}` — a stack of scope frames, live count is
    the size of their union."""
    declare_res = [re.compile(p) for p in cfg.declare]
    decrement_res = [re.compile(p) for p in cfg.decrement]

    stack: List[set] = [set()]
    counts = []
    for idx, line in enumerate(lines):
        start = line_starts[idx]

        for r in decrement_res:
            for m in r.finditer(line):
                if _match_pos_ok(mask, start + m.start(1)):
                    name = m.group(1)
                    for frame in stack:
                        frame.discard(name)

        for r in declare_res:
            for m in r.finditer(line):
                if _match_pos_ok(mask, start + m.start(1)):
                    stack[-1].add(m.group(1))

        counts.append(sum(len(f) for f in stack))

        for k, ch in enumerate(line):
            abs_k = start + k
            if abs_k < len(mask) and mask[abs_k]:
                if ch == '{':
                    stack.append(set())
                elif ch == '}' and len(stack) > 1:
                    stack.pop()

    return counts


def find_states_of_interest(content: str, language: str, project_root: Optional[str] = None,
                             max_points: int = 25) -> List[StatePoint]:
    """Scan `content` for the state(s) with the most live (declared and
    still-in-scope) variables — the "most complex" point(s) to graft a
    continuation into — among lines that are also safe to splice at (real
    code, not mid-string, non-negative running paren depth so we're not
    already inside a broken construct). Ties all come back so
    pick_state_point can still pick among them at random. `project_root` is
    accepted for call-site compatibility but unused — there's no more
    per-project pattern override to load.

    Returns [] for languages with no LIVE_VAR_CONFIGS entry (currently
    just Haskell — see the note above this function's neighboring comment
    block — which always uses pick_state_point(..., lightweight=True)
    instead of calling this at all).
    """
    lang = LANGUAGE_ALIASES.get(language, language)
    cfg = LIVE_VAR_CONFIGS.get(lang)
    if cfg is None:
        return []

    mask = _lexical_mask(content, language)
    lines = content.splitlines()
    line_starts = []
    off = 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1

    counts = (_count_live_flat if cfg.mode == "flat" else _count_live_brace)(
        lines, line_starts, mask, cfg)

    # A line is a safe anchor if it's real code (not mostly inside a
    # string/comment) and not already structurally unbalanced.
    def _is_safe(idx: int, line: str) -> bool:
        start = line_starts[idx]
        real_chars = sum(1 for k in range(start, start + len(line)) if k < len(mask) and mask[k])
        if len(line.strip()) > 0 and real_chars < len(line.strip()) * 0.5:
            return False
        return _paren_depth_at(content, mask, start + len(line)) >= 0

    # First pass: find the true global max among safe lines (must scan the
    # whole file — max_points caps how many *tied* points we return, not
    # how far we look for the maximum).
    best = max((counts[idx] for idx, line in enumerate(lines) if _is_safe(idx, line)),
               default=0)
    if best <= 0:
        return []

    points: List[StatePoint] = []
    for idx, line in enumerate(lines):
        if counts[idx] != best or not _is_safe(idx, line):
            continue
        indent_m = re.match(r"[ \t]*", line)
        points.append(StatePoint(idx, f"live{best}", "", indent_m.group(0)))
        if len(points) >= max_points:
            break
    return points


# ---------------------------------------------------------------------------
# Graft primitive
# ---------------------------------------------------------------------------

_PHP_OPEN_RE = re.compile(r'^\s*<\?php\b')
_PHP_CLOSE_RE = re.compile(r'\?>\s*$')


def strip_donor_wrapper(content: str, language: str) -> str:
    """Strip language-level wrapper tags a donor's *continuation* shouldn't
    carry when it's spliced into a host that's already inside that wrapper
    (e.g. PHP's `<?php ... ?>` — grafting a second `<?php` mid-script is a
    structural break, not a "safe" splice)."""
    lang = LANGUAGE_ALIASES.get(language, language)
    if lang == "php":
        s = _PHP_OPEN_RE.sub("", content, count=1)
        s = _PHP_CLOSE_RE.sub("", s, count=1)
        return s
    return content


def graft_continuation(host_content: str, donor_content: str,
                        host_point: StatePoint, donor_point: Optional[StatePoint],
                        reindent: bool = True) -> str:
    """Splice `donor_content`'s continuation (from `donor_point` onward, or
    its whole body if `donor_point` is None) into `host_content` right
    after `host_point` — grafting one seed's continuation into the other's
    intermediate state, per the state-fusion design. Reindents the donor
    continuation to the host point's indentation when `reindent` is set
    (needed for indentation-sensitive hosts like Python; harmless no-op
    risk for brace languages, so left on by default)."""
    host_lines = host_content.splitlines()
    donor_lines = donor_content.splitlines()

    donor_start = donor_point.line_idx + 1 if donor_point else 0
    # Plain slicing already yields [] when donor_start >= len(donor_lines)
    # (e.g. truncation cut the continuation down to nothing) — do NOT fall
    # back to the whole donor in that case, that would silently re-include
    # everything *before* the point too (including its own container
    # header), which is exactly the kind of structural break this module
    # exists to prevent.
    continuation = donor_lines[donor_start:]

    if reindent and host_point.indent:
        reindented = []
        for ln in continuation:
            stripped = ln.lstrip()
            reindented.append(host_point.indent + stripped if stripped else ln)
        continuation = reindented

    insert_at = min(host_point.line_idx + 1, len(host_lines))
    fused_lines = host_lines[:insert_at] + continuation + host_lines[insert_at:]
    return "\n".join(fused_lines)


def truncate_to_balanced(content: str, start_line_idx: int, language: str) -> int:
    """Return the exclusive end line index for a continuation starting at
    `start_line_idx` such that running paren/bracket/brace depth (relative
    to the depth at the start point) never goes negative — i.e. the
    continuation never emits a stray closing token that would prematurely
    close a scope it didn't open (e.g. donor's own function-closing `}`
    terminating the host function it was grafted into). Brace/paren-based,
    so it's a real safety gate for C-like/MLIR-like languages; indentation-
    based languages (Python) don't need it since graft_continuation's
    reindent already folds the continuation into the host's block instead
    of relying on matched delimiters."""
    lines = content.splitlines()
    if start_line_idx >= len(lines):
        return len(lines)
    mask = _lexical_mask(content, language)
    offsets = []
    off = 0
    for ln in lines:
        offsets.append(off)
        off += len(ln) + 1
    baseline = _paren_depth_at(content, mask, offsets[start_line_idx])
    for i in range(start_line_idx, len(lines)):
        end_off = min(offsets[i] + len(lines[i]), len(mask))
        depth = _paren_depth_at(content, mask, end_off)
        if depth - baseline < 0:
            return i
    return len(lines)


def pick_state_point(content: str, language: str, project_root: Optional[str] = None,
                      cached: Optional[List[dict]] = None,
                      lightweight: bool = False) -> Optional[StatePoint]:
    """Convenience: reuse a cached points list (e.g. seed.metadata['states_of_interest']
    written during --pre-analysis) when available, else compute fresh.

    lightweight (no --pre-analysis): skip the live-variable-count analysis
    (and any cache) entirely and just pick a uniformly random line — no
    per-language config, no scanning, no safety filtering. Deliberately
    the cheapest possible choice of splice point, not a best-effort
    approximation of the real rule.
    """
    import random
    if lightweight:
        lines = content.splitlines()
        if not lines:
            return None
        idx = random.randrange(len(lines))
        indent_m = re.match(r"[ \t]*", lines[idx])
        return StatePoint(idx, "random", "", indent_m.group(0))
    if cached:
        points = [StatePoint.from_dict(d) for d in cached]
    else:
        points = find_states_of_interest(content, language, project_root)
    return random.choice(points) if points else None
