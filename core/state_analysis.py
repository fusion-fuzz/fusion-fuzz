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
   with another seed's continuation. We approximate this with per-language
   regex patterns in three categories called out in the design: resource
   release (`drop()` in Rust, `unset()` in PHP, ...), type conversion, and
   exception-handling boundaries.
2. It must be *safe* to splice at — inserting statements there can't
   break the enclosing syntactic/structural integrity (an unterminated
   string, a line mid-expression, wrong indentation for Python). This
   module checks structural safety; it does not (and does not try to)
   guarantee the grafted program is semantically valid — a donor
   continuation referencing the donor's own earlier variables is exactly
   the kind of "reach into an unfamiliar state" case state fusion is
   for, and it is expected to sometimes fail. The validity-gap metric
   (core/orchestrator.py) tracks how often that happens per language.

Default patterns are intentionally coarse hand-seeded starting points —
core/llm_mapping.py generates/refines a per-project override
(projects/<name>/state_patterns.json) using an LLM plus the validity-gap
signal, per the design's "LLM-Assisted Fusion Adaptation".
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Per-language state-of-interest patterns (regex, matched per source line)
# ---------------------------------------------------------------------------

DEFAULT_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "rust": {
        "resource_release": [r"\bdrop\s*\(", r"\.drop\s*\(\s*\)", r"\bstd::mem::drop\b"],
        "type_conversion":  [r"\bas\s+[A-Za-z_][\w:<>]*", r"\.into\s*\(\s*\)", r"\.try_into\s*\(\s*\)", r"\.unwrap\s*\(\s*\)"],
        "exception_boundary": [r"\bpanic!\s*\(", r"\bResult<", r"\.expect\s*\(", r"\?\s*;"],
    },
    "php": {
        "resource_release": [r"\bunset\s*\(", r"\bfclose\s*\(", r"->__destruct\b"],
        "type_conversion":  [r"\(int\)|\(float\)|\(string\)|\(array\)|\(bool\)", r"\bintval\s*\(", r"\bsettype\s*\("],
        "exception_boundary": [r"\bcatch\s*\(", r"\bfinally\b", r"\bthrow\s+new\b"],
    },
    "cpython": {
        "resource_release": [r"\.close\s*\(\s*\)", r"\bdel\s+\w", r"__exit__\b", r"\bgc\.collect\s*\("],
        "type_conversion":  [r"\bint\s*\(|\bfloat\s*\(|\bstr\s*\(|\blist\s*\(|\bbool\s*\("],
        "exception_boundary": [r"\bexcept\b", r"\bfinally\s*:", r"\braise\b"],
    },
    "go": {
        "resource_release": [r"\bdefer\s+\w", r"\.Close\s*\(\s*\)"],
        "type_conversion":  [r"\w\s*\.\s*\([A-Za-z_][\w.]*\)", r"\bstrconv\.\w+\s*\("],
        "exception_boundary": [r"\brecover\s*\(\s*\)", r"\bpanic\s*\(", r"if\s+err\s*!=\s*nil\b"],
    },
    "clang": {
        "resource_release": [r"\bfree\s*\(", r"~[A-Za-z_]\w*\s*\(", r"\bdelete\s"],
        "type_conversion":  [r"static_cast<|reinterpret_cast<|const_cast<|dynamic_cast<", r"\([A-Za-z_][\w:]*\s*\*?\)\s*\w"],
        "exception_boundary": [r"\bcatch\s*\(", r"\bthrow\b", r"\btry\s*\{"],
    },
    "swift": {
        "resource_release": [r"\bdeinit\b", r"\.close\s*\(\s*\)"],
        "type_conversion":  [r"\bas[!?]?\s+[A-Za-z_]\w*", r"\bAny\b"],
        "exception_boundary": [r"\bcatch\b", r"\bdo\s*\{", r"\btry[!?]?\s"],
    },
    "haskell": {
        "resource_release": [r"\bhClose\b", r"\bfinally\b", r"\bbracket\b"],
        # Deliberately NOT a bare `::\s*[A-Z]\w*` — that also matches an
        # ordinary top-level type *signature* line (`foo :: IO ()`), and
        # grafting right after one (before its `foo = ...` definition) is
        # a structural break, not just a semantic gap. `let x = ... :: T`
        # anchors on the inline-annotation form specifically.
        "type_conversion":  [r"\bfromIntegral\b", r"\brealToFrac\b", r"\blet\s+\S.*::\s*[A-Z]\w*"],
        "exception_boundary": [r"\bcatch\b", r"\bhandle\b", r"\bthrow\b", r"\bthrowIO\b"],
    },
    "flang": {
        "resource_release": [r"(?i)\bCLOSE\s*\(", r"(?i)\bDEALLOCATE\s*\(", r"(?i)\bNULLIFY\s*\("],
        "type_conversion":  [r"(?i)\bINT\s*\(", r"(?i)\bREAL\s*\(", r"(?i)\bDBLE\s*\(", r"(?i)\bTRANSFER\s*\("],
        "exception_boundary": [r"(?i)\bERROR\s+STOP\b", r"(?i)\bSTAT\s*=", r"(?i)\bIOSTAT\s*="],
    },
    # MLIR is a declarative IR, not an executed program, so these categories
    # are an approximation of the paper's runtime-oriented ones: "resource
    # release" maps to explicit dealloc-like ops, "type conversion" to cast
    # ops, and "exception boundary" to the closest thing MLIR has to an
    # error path (assertions / cf branches to an error block).
    "mlir": {
        "resource_release": [r"\bmemref\.dealloc\b", r"\bbufferization\.dealloc\b"],
        "type_conversion":  [r"\barith\.(?:extsi|extui|trunci|bitcast|sitofp|fptosi|uitofp|index_cast)\b",
                              r"\bunrealized_conversion_cast\b"],
        "exception_boundary": [r"\bcf\.assert\b", r"\bcf\.cond_br\b"],
    },
}

# seed metadata "type"/language strings that map onto the keys above.
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


def _load_overrides(project_root: Optional[str]) -> Dict[str, Dict[str, List[str]]]:
    if not project_root:
        return {}
    path = os.path.join(project_root, "state_patterns.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def get_patterns(language: str, project_root: Optional[str] = None) -> Dict[str, List[str]]:
    """Patterns for `language`, with any project-local
    state_patterns.json override merged in (category-by-category union —
    see core/llm_mapping.py, which writes that file)."""
    lang = LANGUAGE_ALIASES.get(language, language)
    base = {k: list(v) for k, v in DEFAULT_PATTERNS.get(lang, {}).items()}
    overrides = _load_overrides(project_root).get(lang, {})
    for category, patterns in overrides.items():
        merged = set(base.get(category, [])) | set(patterns)
        base[category] = sorted(merged)
    return base


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


def find_states_of_interest(content: str, language: str, project_root: Optional[str] = None,
                             max_points: int = 25) -> List[StatePoint]:
    """Scan `content` for state-of-interest points that are also safe to
    splice at (real code, not mid-string, non-negative running paren
    depth so we're not already inside a broken construct)."""
    patterns = get_patterns(language, project_root)
    if not patterns:
        return []

    mask = _lexical_mask(content, language)
    lines = content.splitlines()
    line_starts = []
    off = 0
    for ln in lines:
        line_starts.append(off)
        off += len(ln) + 1

    compiled = {cat: [re.compile(p) for p in pats] for cat, pats in patterns.items()}
    points: List[StatePoint] = []
    for idx, line in enumerate(lines):
        start = line_starts[idx]
        # Skip lines that are entirely/mostly inside a string or comment.
        real_chars = sum(1 for k in range(start, start + len(line)) if k < len(mask) and mask[k])
        if len(line.strip()) > 0 and real_chars < len(line.strip()) * 0.5:
            continue
        for category, pats in compiled.items():
            for pat in pats:
                m = pat.search(line)
                if not m:
                    continue
                abs_pos = start + m.start()
                if abs_pos < len(mask) and not mask[abs_pos]:
                    continue  # match fell inside a string/comment
                depth = _paren_depth_at(content, mask, start + len(line))
                if depth < 0:
                    continue  # already structurally unbalanced before this line — unsafe anchor
                indent_m = re.match(r"[ \t]*", line)
                points.append(StatePoint(idx, category, m.group(0), indent_m.group(0)))
                break
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
                      cached: Optional[List[dict]] = None) -> Optional[StatePoint]:
    """Convenience: reuse a cached points list (e.g. seed.metadata['states_of_interest']
    written during dry-run pre-analysis) when available, else compute fresh."""
    import random
    if cached:
        points = [StatePoint.from_dict(d) for d in cached]
    else:
        points = find_states_of_interest(content, language, project_root)
    return random.choice(points) if points else None
