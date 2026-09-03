"""
core/state_analysis.py — State-Point Profiling & Safe Splice

Implements the "state" half of state fusion: a *state* is the point in a
seed's execution immediately following a given statement. Rather than
fusing only at a seed's final state (what dataflow fusion effectively
does — it links one whole program to another via a single bridge
variable), state fusion selects an *intermediate* program point from each
seed and grafts one seed's continuation into the other's state there —
richer search space, deeper interactions.

Two things make a point usable:

1. It should be a *complex* state — the more live variables (declared,
   and still in scope) a point has, the richer the surface of names a
   donor continuation can collide with, and the more likely the graft
   interacts meaningfully. Unified rule across all supported languages
   except Haskell: every safe point is eligible, and points are sampled
   *proportionally to their live-variable count* (find_state_points
   collects them, pick_state_point / weighted_order draw from them), so
   a below-maximum point is still reachable, just less often. Counts are
   computed per-language via a small (declare/decrement/reset-or-brace-
   scope) regex config below — a static best-effort approximation, not a
   real dataflow/liveness analysis. When no line has any live variables
   at all there is no count to be proportional to, so the weights fall
   back to distance-to-centre (_center_weights: 5 points -> 1, 2, 3, 2,
   1), preferring the middle of the program — a splice near the top has
   almost no preceding state to interact with, one at the very bottom
   leaves nothing to continue into. Haskell instead just
   picks a uniformly random line (HaskellStateFusionStrategy in
   core/fusion.py) unconditionally — its layout-based scoping doesn't fit
   this model, and simplicity won out over trying to make it fit.
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

import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional

# seed metadata "type"/language strings that map onto LIVE_VAR_CONFIGS keys.
LANGUAGE_ALIASES: Dict[str, str] = {
    "python": "cpython", "py": "cpython", "cpython": "cpython",
    "rust": "rust", "rs": "rust",
    "php": "php", "phpt": "php",
    "go": "go",
    "c": "clang", "cpp": "clang", "cxx": "clang", "clang": "clang",
    "objc": "clang", "objcpp": "clang", "m": "clang", "mm": "clang",
    "swift": "swift",
    "haskell": "haskell", "hs": "haskell", "ghc": "haskell",
    "flang": "flang", "fortran": "flang", "f90": "flang",
    "mlir": "mlir",
    "triton": "mlir",
    "naga": "naga", "wgsl": "naga",
    "tint": "naga",
    "javascript": "javascript", "js": "javascript",
    "mjs": "javascript", "v8": "javascript",
    "spidermonkey": "javascript", "sm": "javascript",
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
    # JavaScript scopes `let`/`const` to the enclosing `{}`, so the
    # brace-stack model applies to them exactly. `var` is the exception —
    # it is function-scoped and hoisted, so a var declared inside a block
    # is dropped at that block's `}` when it is really still live. That is
    # a known over-count rather than an oversight: modelling hoisting
    # properly needs a parse, and `var` is the minority form in the mjsunit
    # corpus. No decrement — JavaScript's `delete` removes an object
    # property, not a binding.
    "javascript": LiveVarConfig(
        mode="brace",
        declare=[
            r'\b(?:let|const|var)\s+([A-Za-z_$][\w$]*)',
            r'\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)',
            r'\bclass\s+([A-Za-z_$][\w$]*)',
            r'\bfor\s*\(\s*(?:let|const|var)\s+([A-Za-z_$][\w$]*)',
            r'\bcatch\s*\(\s*([A-Za-z_$][\w$]*)\s*\)',
        ],
        multi_name=False,
    ),
    # Go scopes names to their enclosing `{}`, so the brace-stack model
    # applies. `:=` is the dominant declaration form by a wide margin;
    # `var` and the range/type-switch binders cover the rest. No decrement:
    # Go has no `del`/`free` that ends a name's scope early — an unused one
    # is a compile error instead, which is a fact about validity, not
    # liveness.
    "go": LiveVarConfig(
        mode="brace",
        declare=[
            r'\bvar\s+([A-Za-z_]\w*)',
            r'^\s*([A-Za-z_]\w*)\s*:=(?!=)',
            r'\bfor\s+([A-Za-z_]\w*)\s*(?::=|,)',
            r'\brange\s+([A-Za-z_]\w*)\b',
            r'\bswitch\s+([A-Za-z_]\w*)\s*:=\s*\w+\.\(type\)',
        ],
        multi_name=False,
    ),
    # MLIR is SSA: a %value's scope is exactly its enclosing region, so the
    # brace-stack model is exact here, not a heuristic — no decrement
    # needed since SSA values are never undefined once in scope.
    "mlir": LiveVarConfig(
        mode="brace",
        declare=[r'(%[A-Za-z0-9_$.]+)\s*='],
    ),
    "naga": LiveVarConfig(
        mode="brace",
        declare=[
            r'\b(?:var(?:\s*<[^>]+>)?|let|const|override)\s+([A-Za-z_]\w*)',
            r'\bfn\s+\w+\s*\([^)]*\b([A-Za-z_]\w*)\s*:',
        ],
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
    # Template literals make the backtick a string delimiter in JS, and
    # regex literals are deliberately not modelled: telling `/` division
    # from a regex needs parse context, and treating it as a quote would
    # swallow arbitrary code.
    "javascript": {"line": ["//"], "block": [("/*", "*/")],
                   "quotes": ['"', "'", "`"]},
    "clang":   {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"', "'"]},
    "swift":   {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"']},
    "haskell": {"line": ["--"], "block": [("{-", "-}")], "quotes": ['"']},
    # Fortran uses doubled-quote escaping ('' / "") rather than backslash
    # escaping; the generic backslash-aware scanner below is a reasonable
    # approximation (worst case: it closes a string one quote earlier than
    # a doubled escape would), not exact — fine for a heuristic safety gate.
    "flang":   {"line": ["!"], "block": [], "quotes": ['"', "'"]},
    "mlir":    {"line": ["//"], "block": [], "quotes": ['"']},
    "naga":    {"line": ["//"], "block": [("/*", "*/")], "quotes": ['"']},
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


def _paren_depth_prefix(content: str, mask: List[bool]) -> List[int]:
    """Running paren/bracket/brace depth after each prefix of `content` —
    prefix[i] is the depth after content[:i]. Built once in O(n).

    Both callers (find_state_points's _is_safe, and
    truncate_to_balanced) probe the depth at O(n) increasing offsets while
    walking a file line by line. The previous approach called a
    _paren_depth_at(content, mask, upto) helper that rescanned from
    position 0 on every probe, making a single find_state_points /
    truncate_to_balanced call O(n^2) in content length — cheap on typical
    seeds but a multi-minute stall (and, since this runs inside a
    ThreadPoolExecutor worker holding the GIL, a full stall of every
    worker thread) on the corpus's larger outliers. Precomputing the
    prefix once turns every probe into an O(1) lookup."""
    n = len(content)
    prefix = [0] * (n + 1)
    depth = 0
    for i in range(n):
        if mask[i] and content[i] in "([{":
            depth += 1
        elif mask[i] and content[i] in ")]}":
            depth -= 1
        prefix[i + 1] = depth
    return prefix


# ---------------------------------------------------------------------------
# State point discovery
# ---------------------------------------------------------------------------

_LIVE_CATEGORY_RE = re.compile(r"^live(\d+)$")


@dataclass
class StatePoint:
    line_idx: int            # 0-based line index whose END is the state
    category: str
    matched_text: str
    indent: str = ""
    live_count: int = 0      # sampling weight: live variables at this point

    def to_dict(self) -> dict:
        return {"line_idx": self.line_idx, "category": self.category,
                "matched_text": self.matched_text, "indent": self.indent,
                "live_count": self.live_count}

    @staticmethod
    def from_dict(d: dict) -> "StatePoint":
        live = d.get("live_count")
        if live is None:
            # Metadata cached before live_count existed. The count was
            # encoded in the category ("live7"), so recover it from there
            # and fall back to 0 (which pick_state_point treats as
            # unweighted) only when even that is absent.
            m = _LIVE_CATEGORY_RE.match(str(d.get("category", "")))
            live = int(m.group(1)) if m else 0
        return StatePoint(d["line_idx"], d["category"], d.get("matched_text", ""),
                          d.get("indent", ""), int(live))


# Haskell is deliberately NOT in LIVE_VAR_CONFIGS and has no dedicated
# find_state_points path: its real "many variables" equivalent
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


def _center_weights(n: int) -> List[int]:
    """Triangular weights over n positions, peaking in the middle:
    n=5 -> [1, 2, 3, 2, 1]; n=4 -> [1, 2, 2, 1].

    Used when live-variable counts give nothing to weight by. The middle
    of a program is the better place to splice on structural grounds
    alone: a point near the top has almost no preceding state for the
    donor continuation to interact with, and one at the very bottom
    leaves the host almost nothing to continue into. Weighting by
    distance-to-centre keeps the extremes reachable while preferring the
    middle, instead of treating every line as equally good.
    """
    return [min(i + 1, n - i) for i in range(n)]


def _sampling_weights(points: List[StatePoint]) -> List[int]:
    """Weight per point: its live-variable count, or — when no point has
    any live variable at all — its distance-to-centre weight."""
    weights = [max(0, p.live_count) for p in points]
    return weights if any(weights) else _center_weights(len(points))


def weighted_order(points: List[StatePoint],
                   rng: Optional[random.Random] = None) -> List[StatePoint]:
    """`points` permuted so that each point's chance of coming first is
    proportional to its live-variable count.

    Uses the Efraimidis-Spirakis key u**(1/w): sorting by that key
    descending is exactly weighted sampling without replacement, so the
    whole permutation — not just its head — is weighted. Callers that
    scan for the first *usable* point (MLIR's donor-point search) get the
    same proportional preference as callers that just take the first.

    Points with no live variables keep a chance of being reached (they
    sort last) but are never preferred; when *every* count is zero there
    is nothing to be proportional to and _center_weights takes over, so
    the order prefers points near the middle of the program.
    """
    rng = rng or random
    if not points:
        return []
    weights = _sampling_weights(points)
    keyed = []
    for point, weight in zip(points, weights):
        # rng.random() is in [0,1); the u==0 corner would make every
        # positive weight tie at 0.0, so nudge it into (0,1].
        key = (rng.random() or 1e-12) ** (1.0 / weight) if weight > 0 else -1.0
        keyed.append((key, point))
    keyed.sort(key=lambda kp: kp[0], reverse=True)
    return [point for _, point in keyed]


def find_state_points(content: str, language: str, project_root: Optional[str] = None,
                      max_points: int = 100,
                      rng: Optional[random.Random] = None) -> List[StatePoint]:
    """Scan `content` for every safe-to-splice state point, each tagged
    with its live-variable count so callers can sample *proportionally*
    to that count (see pick_state_point / weighted_order).

    Every line is a candidate — there is no up-front "is this line safe"
    filter; where the text can actually be cut is decided separately by
    segment_boundaries. Every line is returned — points below the file's maximum live-variable
    count are eligible too, just less likely to be drawn, and when the
    file has no live variables at all every safe line comes back with
    weight 0 so the caller's distance-to-centre fallback picks among
    them. `project_root`
    is accepted for call-site compatibility but unused — there's no more
    per-project pattern override to load.

    At most `max_points` come back. The cap exists to bound the cached
    metadata: without one, a 9000-line seed caches 9000 points and the
    clang corpus.db grows to ~488 MB. 100 is comfortably above the
    corpus's median (26 points/seed), so only the long tail is trimmed.

    When there are more, the list is cut down by a *uniform* random
    subsample rather than by keeping the highest counts or the first N.
    Uniform is the right choice precisely because the caller weights
    afterwards: it preserves the relative weights among the survivors,
    whereas subsampling by weight would apply the weighting twice and
    positional truncation would bias the cache toward the top of the file.

    When there's at least one safe line but none of them declare any live
    variable (best == 0 — e.g. a trivial file, or one whose declare/
    decrement patterns just never matched), falls back to a single
    uniformly-random safe line instead of coming back empty, so state
    fusion still gets a splice point rather than degenerating to whatever
    fixed fallback the caller applies for []. Only returns [] when there
    is no safe line at all (empty content, or every line is unsafe).

    Returns [] outright for languages with no LIVE_VAR_CONFIGS entry
    (currently just Haskell — see the note above this function's
    neighboring comment block — which always uses
    pick_state_point(..., lightweight=True) instead of calling this at
    all).
    """
    rng = rng or random
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

    # Every line is a candidate anchor. There used to be an "is this line
    # safe to anchor at" filter here (real code rather than string/comment
    # interior, and non-negative running delimiter depth); it was dropped
    # deliberately. Structural integrity of the *result* is enforced where
    # the text is actually divided — segment_boundaries, which requires
    # depth exactly zero — and an anchor landing somewhere odd produces an
    # ill-formed child, which is a legitimate thing to feed a compiler
    # rather than something to filter out up front.
    safe_idxs = list(range(len(lines)))
    if not safe_idxs:
        return []

    best = max(counts[idx] for idx in safe_idxs)

    points: List[StatePoint] = []
    for idx in safe_idxs:
        indent_m = re.match(r"[ \t]*", lines[idx])
        # best <= 0: nothing in this file declares a live variable, so
        # every safe line is returned with weight 0 and the caller's
        # _center_weights fallback decides between them — preferring the
        # middle of the program. (Previously this branch collapsed to a
        # single pre-chosen line, which meant a cached seed spliced at
        # the very same line on every fusion for the rest of its life.)
        category = f"live{counts[idx]}" if best > 0 else "nolive"
        points.append(StatePoint(idx, category, "",
                                 indent_m.group(0), counts[idx]))
    if len(points) > max_points:
        points = sorted(rng.sample(points, max_points), key=lambda p: p.line_idx)
    return points


# Pre-rename alias. The old name described the old behaviour (return only
# the maximum-live-variable points); selection is proportional now, so the
# name is kept only so out-of-tree callers don't break.
find_most_complex_state = find_state_points


# ---------------------------------------------------------------------------
# Graft primitive
# ---------------------------------------------------------------------------

_PHP_OPEN_RE = re.compile(r'^\s*<\?php\b')
_PHP_CLOSE_RE = re.compile(r'\?>\s*$')


def graft_continuation(host_content: str, donor_content: str,
                        host_point: StatePoint, donor_point: Optional[StatePoint],
                        reindent: bool = True, tag_comment: Optional[str] = None) -> str:
    """Splice `donor_content`'s continuation (from `donor_point` onward, or
    its whole body if `donor_point` is None) into `host_content` right
    after `host_point` — grafting one seed's continuation into the other's
    intermediate state, per the state-fusion design. Reindents the donor
    continuation to the host point's indentation when `reindent` is set
    (needed for indentation-sensitive hosts like Python; harmless no-op
    risk for brace languages, so left on by default).

    `tag_comment`, when given (a full comment, e.g. "// state fusion", in
    the host language's own syntax — callers own picking that syntax), is
    appended as a trailing same-line comment onto the first non-blank line
    of the grafted continuation — deliberately NOT its own standalone
    comment line, so a line-based test-case reducer can't strip the tag
    without also removing the statement it marks. Only applied when
    there's actually a continuation to graft, so an empty splice doesn't
    leave a dangling comment with nothing under it."""
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

    if tag_comment and continuation:
        idx = next((i for i, ln in enumerate(continuation) if ln.strip()), 0)
        continuation[idx] = continuation[idx] + f"  {tag_comment}"

    insert_at = min(host_point.line_idx + 1, len(host_lines))
    fused_lines = host_lines[:insert_at] + continuation + host_lines[insert_at:]
    return "\n".join(fused_lines)


@dataclass
class FusionSegments:
    """The four pieces a state fusion is composed from, plus where each
    input was cut. `a_split`/`b_split` are exclusive end indices of the
    respective prefixes, i.e. prefix == lines[:split], suffix == lines[split:].
    """
    a_prefix: List[str]
    a_suffix: List[str]
    b_prefix: List[str]
    b_suffix: List[str]
    a_split: int
    b_split: int
    a_point: int
    b_point: int

    def to_metadata(self) -> dict:
        """Segment ranges for the fused Seed's metadata, so a saved
        reproducer records exactly how it was assembled."""
        return {
            "a_point": self.a_point, "b_point": self.b_point,
            "a_split": self.a_split, "b_split": self.b_split,
            "segment_lines": {
                "a_prefix": len(self.a_prefix), "b_prefix": len(self.b_prefix),
                "a_suffix": len(self.a_suffix), "b_suffix": len(self.b_suffix),
            },
        }


@lru_cache(maxsize=1024)
def _segment_boundaries_cached(content: str, language: str) -> tuple:
    return tuple(_compute_segment_boundaries(content, language))


def segment_boundaries(content: str, language: str) -> List[int]:
    """Line indices after which `content` can be cut into two independently
    well-formed statement sequences. Memoized — see
    _compute_segment_boundaries for the rule and the cost this avoids."""
    return list(_segment_boundaries_cached(content, language))


# Keywords that *continue* a compound statement rather than starting one.
# A cut immediately before any of them separates the clause from its
# header, which is a syntax error in every language here — Python's
# `elif`/`else`/`except`/`finally`/`case`, and the brace languages'
# `else`/`catch`/`finally`/`while` (the tail of a do-while).
_CONTINUATION_RE = re.compile(
    r'^(?:\}\s*)?(?:elif|else|except|finally|case|catch|while)\b'
    r'|^\}\s*(?:else|catch|finally|while)?\s*[;{]?\s*$')


def _compute_segment_boundaries(content: str, language: str) -> List[int]:
    """Line indices after which `content` can be cut into two independently
    well-formed statement sequences.

    A cut is legal only where the running delimiter depth is back to zero —
    otherwise the prefix leaves an unclosed brace/paren that the suffix,
    now separated from it by another program's text, no longer closes. For
    indentation-scoped languages (Python) the next non-blank line must also
    start at column zero, or the cut would separate a compound statement's
    header from its body.

    This is *additional* to find_state_points' eligibility filtering, not a
    replacement for it: that decides which points are interesting, this
    decides where the text can actually be divided.
    """
    lines = content.splitlines()
    if not lines:
        return []
    mask = _lexical_mask(content, language)
    offsets, off = [], 0
    for ln in lines:
        offsets.append(off)
        off += len(ln) + 1
    depth = _paren_depth_prefix(content, mask)
    lang = LANGUAGE_ALIASES.get(str(language).lower(), language)
    indentation_scoped = lang == "cpython"

    def _next_nonblank(i):
        for j in range(i + 1, len(lines)):
            if lines[j].strip():
                return lines[j]
        return None

    out = []
    for i, line in enumerate(lines):
        end = min(offsets[i] + len(line), len(depth) - 1)
        if depth[end] != 0:
            continue
        # The cut must not land inside a string or comment. `mask` already
        # records that (it is what keeps parens inside strings out of the
        # depth count), but it was only ever consulted for the depth.
        #
        # A triple-quoted string does not change paren depth and its prose
        # is often unindented, so every line of a docstring or of embedded
        # test data satisfied both tests above and was offered as a legal
        # boundary. Cutting there ends the prefix mid-literal and leaves
        # the suffix's prose as bare code — "unterminated string literal",
        # or an email header parsed as an expression. On the CPython corpus
        # this was the whole of the remaining syntax failures once
        # continuation clauses were handled.
        if end < len(mask) and not mask[end]:
            continue
        if indentation_scoped:
            nxt = _next_nonblank(i)
            if nxt is not None and nxt[:1].isspace():
                continue
        # A continuation clause is at the same indentation (and, for brace
        # languages, the same delimiter depth) as the header it belongs to,
        # so neither test above rejects it — yet cutting here severs
        # `elif`/`else`/`except` from the `if`/`try` that owns it. On the
        # CPython corpus this was 58 of 66 remaining parse failures: the
        # four-segment interleave kept splitting top-level if/elif chains.
        nxt = _next_nonblank(i)
        if nxt is not None and _CONTINUATION_RE.match(nxt.lstrip()):
            continue
        out.append(i)
    return out


def split_at_point(content: str, point: Optional[StatePoint], language: str,
                   boundaries: Optional[List[int]] = None):
    """Cut `content` into (prefix, suffix, split_idx) around `point`.

    The point's own line goes to the *prefix*: a StatePoint denotes the
    state immediately **after** that line has executed (see StatePoint's
    docstring), so the prefix is exactly the code that produced that state
    and the suffix is the continuation from it.

    When the point does not sit on a legal boundary (see
    segment_boundaries) the cut moves to the nearest legal line at or
    after it, so the state the point represents is still contained in the
    prefix; failing that, to the nearest one before it. Returns None when
    `content` has no legal boundary at all.
    """
    lines = content.splitlines()
    if not lines:
        return None
    # `boundaries` comes from --pre-analysis when available: computing it
    # means two O(n) character scans (_lexical_mask + _paren_depth_prefix)
    # per seed per fusion, ~20% of the fusion loop's Python time. Line
    # numbering survives both mutation and include hoisting (measured at
    # 100% on the clang corpus), so an index computed once up front stays
    # valid for every later fusion of that seed.
    if boundaries is None:
        boundaries = segment_boundaries(content, language)
    if not boundaries:
        return None

    target = point.line_idx if point else len(lines) - 1
    at_or_after = [b for b in boundaries if b >= target]
    chosen = at_or_after[0] if at_or_after else boundaries[-1]

    split = chosen + 1
    return lines[:split], lines[split:], split


def interleave_segments(a_content: str, b_content: str,
                        a_point: Optional[StatePoint], b_point: Optional[StatePoint],
                        language: str, tag_comment: Optional[str] = None,
                        a_boundaries: Optional[List[int]] = None,
                        b_boundaries: Optional[List[int]] = None):
    """Four-segment state fusion: A_prefix + B_prefix + A_suffix + B_suffix.

    Both programs are cut at their own fusion point and *all four* pieces
    appear in the result, each keeping its own internal statement order.
    A's prefix runs first, then B's prefix builds B's state on top of it,
    then A's continuation resumes — now with B's names in scope — and
    finally B's continuation runs against whatever A left behind. Every
    segment therefore executes in a state the other program produced,
    which is the point of fusing them.

    This replaces the earlier suffix-grafting scheme (A_prefix + B_suffix +
    A_suffix), which dropped B's prefix entirely: B's continuation ran
    without the code that set up the state it expects, and B's own setup
    never interacted with A at all.

    Returns (fused_text, FusionSegments) or None when either side has no
    legal split boundary.
    """
    a_split_res = split_at_point(a_content, a_point, language, a_boundaries)
    b_split_res = split_at_point(b_content, b_point, language, b_boundaries)
    if a_split_res is None or b_split_res is None:
        return None

    a_prefix, a_suffix, a_split = a_split_res
    b_prefix, b_suffix, b_split = b_split_res

    def _tagged(seg):
        """Mark the first real line of a B segment, so a reader (and a
        line-based reducer) can see which statements came from B."""
        if not tag_comment or not seg:
            return seg
        seg = list(seg)
        idx = next((i for i, ln in enumerate(seg) if ln.strip()), None)
        if idx is not None:
            seg[idx] = seg[idx] + f"  {tag_comment}"
        return seg

    fused_lines = (list(a_prefix) + _tagged(b_prefix)
                   + list(a_suffix) + _tagged(b_suffix))
    segments = FusionSegments(
        a_prefix=a_prefix, a_suffix=a_suffix,
        b_prefix=b_prefix, b_suffix=b_suffix,
        a_split=a_split, b_split=b_split,
        a_point=a_point.line_idx if a_point else -1,
        b_point=b_point.line_idx if b_point else -1,
    )
    return "\n".join(fused_lines), segments


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
    paren_prefix = _paren_depth_prefix(content, mask)
    baseline = paren_prefix[offsets[start_line_idx]]
    for i in range(start_line_idx, len(lines)):
        end_off = min(offsets[i] + len(lines[i]), len(mask))
        depth = paren_prefix[end_off]
        if depth - baseline < 0:
            return i
    return len(lines)


def pick_state_point(content: str, language: str, project_root: Optional[str] = None,
                      cached: Optional[List[dict]] = None,
                      lightweight: bool = False,
                      rng: Optional[random.Random] = None) -> Optional[StatePoint]:
    """Draw one state point, sampled *proportionally to its live-variable
    count* among all eligible points — a point with 6 live variables is
    drawn twice as often as one with 3, and a below-maximum point is
    still drawn, just less often. Reuses a cached points list (e.g.
    seed.metadata['most_complex_states'] written during --pre-analysis)
    when available, else computes fresh.

    Zero-weight fallback: when no eligible point has any live variable
    (a trivial file, or a stale cache whose entries predate live_count
    and carry no recoverable category), there is nothing to be
    proportional to, so the weights come from _center_weights instead —
    points near the middle of the program are preferred, the first and
    last lines least (5 points -> weights 1, 2, 3, 2, 1).

    `rng` defaults to the `random` module, so a caller that seeds it
    (`random.seed(...)`) still gets deterministic selection; pass an
    explicit random.Random for an independent stream.

    lightweight (no --pre-analysis): skip the live-variable-count analysis
    (and any cache) entirely and just pick a uniformly random line — no
    per-language config, no scanning, no safety filtering. Deliberately
    the cheapest possible choice of splice point, not a best-effort
    approximation of the real rule.

    `cached` uses `is not None` rather than truthiness: an explicit `[]`
    means pre-analysis already ran and confirmed zero state points for
    this content, which is a cache hit (no points, don't recompute) —
    treating it the same as "never analyzed" (key absent, `None`) would
    force a full rescan on every fusion attempt for exactly the seeds
    that always come up empty.
    """
    rng = rng or random
    if lightweight:
        lines = content.splitlines()
        if not lines:
            return None
        idx = rng.randrange(len(lines))
        indent_m = re.match(r"[ \t]*", lines[idx])
        return StatePoint(idx, "random", "", indent_m.group(0), 0)
    if cached is not None:
        points = [StatePoint.from_dict(d) for d in cached]
    else:
        points = find_state_points(content, language, project_root, rng=rng)
    if not points:
        return None
    return rng.choices(points, weights=_sampling_weights(points), k=1)[0]
