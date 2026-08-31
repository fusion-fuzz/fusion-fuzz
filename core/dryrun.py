"""
core/dryrun.py — Pre-Fuzzing Seed Execution: Validity Filter + Metadata Collector

Two independent concerns share one execution pass over the corpus, each
gated by its own CLI flag (main.py):

  --dry-run       execute each seed once, keep only rc==0 ("is this seed
                  even valid on its own?"). No metadata collection.
  --pre-analysis  execute each seed once (probe-instrumented, for
                  languages that need one) and collect *static*
                  (source-level regex analysis) and *dynamic*
                  (stdout/stderr) metadata for the fusion strategies —
                  dataflow graphs, declared/observed types, and
                  core/state_analysis.py's live-variable states of
                  interest. Does not filter the corpus.

Pass both and they share the same single execution per seed rather than
running it twice — run_dryrun_with_metadata's collect_metadata/
filter_valid params are what --pre-analysis/--dry-run map to.

Why this matters for fusion quality
------------------------------------
The fusion strategies need to know what *types* the variables in a seed
carry so they can build valid bridge expressions (e.g. only bridge an
`i32` from seed A into a position that expects a numeric type in seed B).
Without this information the bridge is a random guess and will fail type
checking for strict languages like Rust, Swift, or Go.

Which keys are actually consumed
------------------------------------
Only four of these are read at fusion time; the rest are collected but
have no consumer today (their original reader, the producer-consumer
matcher in core/resource_matching.py, was removed):

  most_complex_states  -> every *StateFusionStrategy, on every fusion
  segment_boundaries   -> four-segment state fusion's split points
  has_declaration      -> Clang/Naga declaration fusion's is_viable_pair
  rc / dryrun_done     -> --dry-run's corpus filter

Everything else (var_types, struct_names, functions, imports, classes,
top_level_vars, undefined_refs, primitive_vars, cloneable_vars,
fn_signatures, complexity_score, has_generics/lifetimes/unsafe,
types_used, constants, structs, dynamic_types, live_vars, line_count,
byte_size) is written and never read back. They are kept rather than
deleted because they are cheap and plausibly useful — measured on the
clang corpus they are 6 B/seed, 0.1% of stored metadata, against
most_complex_states' 4.5 KB. Delete a collector's field only if you have
first checked it is still unread *and* that it costs something.

Metadata keys written per language
------------------------------------
Rust:
  var_types        dict[str,str]   variable name → declared type string
  primitive_vars   list[str]       names with Copy/primitive types
  cloneable_vars   list[str]       names with Clone-able types
  fn_signatures    list[str]       function signatures (non-main)
  struct_names     list[str]       struct identifiers
  has_lifetimes    bool
  has_generics     bool
  has_unsafe       bool
  complexity_score int
  line_count       int

Python / CPython:
  top_level_vars   list[str]       top-level assigned variable names
  dynamic_types    dict[str,str]   name → observed runtime type (compatible
                                   with CPythonFusionStrategy)
  functions        list[str]
  classes          list[str]
  imports          list[str]
  line_count       int

Go:
  var_types        dict[str,str]
  struct_names     list[str]
  has_generics     bool
  line_count       int

MLIR:
  functions        list[str]
  constants        list[dict]
  types_used       list[str]
  line_count       int

WGSL / naga / wgslc:
  functions        list[str]
  structs          list[str]
  var_types        dict[str,str]
  line_count       int

Clang (C / C++ / Obj-C):
  has_declaration  bool            whether the seed has a struct/class/enum
                                   ClangDeclarationFusionStrategy could
                                   donate — see is_viable_pair in
                                   core/fusion.py

All languages also receive (whenever the seed is executed at all, under
either flag):
  dryrun_done        bool          marker so a later --dry-run run can
                                    skip re-executing this seed
  rc                 int           return code from the execution

--pre-analysis additionally sets:
  pre_analysis_done  bool          marker so a later --pre-analysis run
                                    can skip re-collecting this seed
  most_complex_states list[dict]   core/state_analysis.py's StatePoint
                                    cache (each entry carries live_count,
                                    the sampling weight), consumed by
                                    *StateFusionStrategy
"""

import json
import logging
import re
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

logger = logging.getLogger("FFL.DryRun")


# ---------------------------------------------------------------------------
# Rust type sets
# ---------------------------------------------------------------------------

_RUST_COPY_TYPES: frozenset = frozenset({
    "i8", "i16", "i32", "i64", "i128", "isize",
    "u8", "u16", "u32", "u64", "u128", "usize",
    "f32", "f64", "bool", "char",
})

_RUST_CLONE_TYPES: frozenset = frozenset({
    "String", "Vec", "HashMap", "HashSet", "BTreeMap", "BTreeSet",
    "Option", "Result", "Box", "Rc", "Arc", "PathBuf",
})


# ---------------------------------------------------------------------------
# Base collector
# ---------------------------------------------------------------------------

# Metadata keys --pre-analysis leaves on *every* seed whose language
# core/state_analysis.py knows about. Listed here so a seed analysed by an
# older build — one that predates a key, or wrote it under a name since
# retired — is re-analysed instead of being skipped forever on the strength
# of its pre_analysis_done flag alone.
PRE_ANALYSIS_KEYS = ("most_complex_states", "segment_boundaries")

# Bumped whenever the *meaning* of what --pre-analysis writes changes, as
# opposed to which keys it writes. A seed stamped with an older version is
# re-analysed even though every key is present — without this, a change
# like "state points are no longer capped at 25 per seed" leaves the whole
# corpus on the old, narrower cache forever, since the key it lives under
# never changed. History:
#   1  state points capped at 25/seed; anchors filtered by an _is_safe gate
#   2  cap removed (every line is a candidate anchor), _is_safe gate dropped
#   3  cap reinstated at 100/seed — uncapped pushed corpus.db to 488 MB on
#      the long tail (one seed cached 9230 points) for no measured gain
# Bumped twice while fixing core/state_analysis._compute_segment_boundaries,
# which now rejects two kinds of cut it used to offer:
#   4: immediately before a continuation clause (elif/else/except/finally/
#      catch), which severs the clause from the header that owns it.
#   5: anywhere inside a string or comment. A triple-quoted string changes
#      no paren depth and its prose is often unindented, so every docstring
#      line looked like a legal boundary.
# A stale cache is worse than none here: it silently overrides the
# corrected computation with the indices the old one produced.
PRE_ANALYSIS_VERSION = 5

# Keys written by builds whose semantics no longer match what the code
# reads. states_of_interest held only the maximum-live-variable points
# (core/state_analysis.py now returns every safe point, each carrying the
# live_count it is sampled by), so reading it back would silently restore
# the maximum-only selection it was built with. Dropped on rewrite.
OBSOLETE_METADATA_KEYS = ("states_of_interest",)


class BaseMetadataCollector:
    """
    Collects static and optionally dynamic metadata from one seed execution.

    Subclasses override:
        static_collect(content, filename)  → dict
        dynamic_collect(content, result)   → dict   (optional)
        provides                           → tuple  (keys static/dynamic
                                             _collect are expected to set,
                                             checked for cache freshness)

    The result object has .return_code, .stdout, .stderr attributes.
    """

    #: keys this collector guarantees to write; see PRE_ANALYSIS_KEYS
    provides: tuple = ()

    language: str = "generic"

    def static_collect(self, content: str, filename: str = "") -> dict:
        """Extract metadata purely from source text — no execution needed."""
        return {}

    def dynamic_collect(self, content: str, result, filename: str = "") -> dict:
        """
        Extract metadata from the execution result.
        Override for languages where stdout carries useful information.
        """
        return {}

    def instrument_for_probe(self, content: str) -> str:
        """
        Optionally transform seed content before the dry-run execution so
        that execution output carries richer information.  The default
        implementation is a no-op.
        """
        return content

    def collect(self, seed, result, fallback_language: Optional[str] = None) -> dict:
        """Combine static + dynamic metadata into one dict.

        `fallback_language` is the project's own language, used when the
        seed's `type` names none — e.g. --bug-corpus injects seeds tagged
        `bug_corpus`, which is a provenance marker, not a language. Without
        the fallback those seeds get no state-point or boundary cache at
        all and every fusion re-analyses them from scratch (they are half
        the clang corpus).
        """
        content = seed.content or ""
        filename = (seed.metadata or {}).get("filename", "")

        meta: dict = {}

        try:
            meta.update(self.static_collect(content, filename))
        except Exception as e:
            logger.debug(f"[{self.language}] static_collect error for {seed.id}: {e}")

        if result is not None:
            try:
                meta.update(self.dynamic_collect(content, result, filename))
            except Exception as e:
                logger.debug(f"[{self.language}] dynamic_collect error for {seed.id}: {e}")

        # Most-complex-state pre-analysis (state fusion design, core/
        # state_analysis.py): computed once here during the one-time
        # dry-run pass and cached in seed metadata, so state fusion never
        # has to re-scan a seed at fusion time. Language is derived from
        # the seed's own "type" metadata (falls back to this collector's
        # language) via state_analysis.LANGUAGE_ALIASES; a no-op for
        # languages without patterns defined yet.
        try:
            from .state_analysis import (find_state_points, segment_boundaries,
                                         LANGUAGE_ALIASES)
            seed_type = (seed.metadata or {}).get("type") or self.language
            lang = (LANGUAGE_ALIASES.get(str(seed_type).lower())
                    or LANGUAGE_ALIASES.get(str(fallback_language or "").lower()))
            if lang:
                # Where the text may be cut into independently well-formed
                # halves (four-segment state fusion). Two O(n) character
                # scans, measured at ~20% of the fusion loop's Python time
                # when recomputed per fusion. Line numbering survives both
                # mutation and include hoisting, so indices computed here
                # stay valid for every later fusion of this seed.
                meta["segment_boundaries"] = segment_boundaries(content, lang)
                points = find_state_points(content, lang)
                # Always record the result, even when empty — pick_state_point
                # distinguishes "cached: no points" ([]) from "never analyzed"
                # (key absent) to decide whether to recompute at fusion time.
                # Guarding this on `if points` used to drop the key for seeds
                # with zero state points, making them silently re-run the full
                # scan on every single fusion attempt for the seed's entire
                # lifetime instead of caching the (empty) answer once here.
                meta["most_complex_states"] = [p.to_dict() for p in points]
        except Exception as e:
            logger.debug(f"[{self.language}] most-complex-state pre-analysis error for {seed.id}: {e}")

        # Both stamps belong here, next to the analysis they describe:
        # collect() is only called when metadata is actually being
        # collected, and a caller that had to remember to set
        # pre_analysis_done separately made collect()'s own output fail
        # _metadata_is_fresh — i.e. the two halves of the contract could
        # drift apart unnoticed.
        meta["pre_analysis_done"] = True
        meta["pre_analysis_version"] = PRE_ANALYSIS_VERSION
        meta["dryrun_done"] = True
        if result is not None:
            meta["rc"] = result.return_code

        return meta


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

class RustMetadataCollector(BaseMetadataCollector):
    """
    Static analysis of Rust seeds.

    Rust's mandatory type annotations make regex-based extraction reliable:
        let [mut] name: Type = ...
    We categorise each variable as Copy-safe (direct bridge), Clone-able
    (bridge with .clone()), or complex (avoid bridging without LLM help).
    """

    language = "rust"

    _LET_TYPE = re.compile(
        r'\blet\s+(?:mut\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=;\n]+?)\s*(?:=|;)',
    )
    _FN_SIG = re.compile(
        r'\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)'
        r'(?:\s*->\s*([^{;]+))?',
    )
    _STRUCT = re.compile(r'\bstruct\s+([a-zA-Z_][a-zA-Z0-9_]*)')
    _LIFETIME = re.compile(r"'[a-zA-Z_][a-zA-Z0-9_]*")
    _GENERIC  = re.compile(r'<\s*[A-Z][a-zA-Z0-9_]*(?:\s*:\s*[^,>]+)?(?:\s*,\s*[A-Z][a-zA-Z0-9_]*(?:\s*:\s*[^,>]+)?)*\s*>')

    def static_collect(self, content: str, filename: str = "") -> dict:
        meta: dict = {}

        # 1. Variable type annotations
        var_types: Dict[str, str] = {}
        for m in self._LET_TYPE.finditer(content):
            name = m.group(1)
            raw_type = m.group(2).strip()
            # Normalise: strip leading & / &mut / lifetime refs
            norm = re.sub(r"&(?:'[a-z_]+\s+)?(?:mut\s+)?", "", raw_type).strip()
            var_types[name] = norm
        meta["var_types"] = var_types

        # 2. Copy / primitive variables — ideal bridge sources (no clone needed)
        meta["primitive_vars"] = [
            name for name, t in var_types.items()
            if self._base_type(t) in _RUST_COPY_TYPES
        ]

        # 3. Clone-able variables — bridgeable if we emit .clone()
        meta["cloneable_vars"] = [
            name for name, t in var_types.items()
            if self._base_type(t) in (_RUST_COPY_TYPES | _RUST_CLONE_TYPES)
        ]

        # 4. Function signatures (skip main)
        fn_sigs: List[str] = []
        for m in self._FN_SIG.finditer(content):
            if m.group(1) != "main":
                fn_sigs.append(m.group(0).strip()[:200])
        meta["fn_signatures"] = fn_sigs

        # 5. Struct names
        meta["struct_names"] = list(dict.fromkeys(self._STRUCT.findall(content)))

        # 6. Lifetime presence (excluding 'static)
        lifetimes = [l for l in self._LIFETIME.findall(content) if l != "'static"]
        meta["has_lifetimes"] = len(lifetimes) > 0

        # 7. Generic type parameter presence
        meta["has_generics"] = bool(self._GENERIC.search(content))

        # 8. Unsafe blocks
        meta["has_unsafe"] = bool(re.search(r'\bunsafe\s*\{', content))

        # 9. Complexity score
        meta["line_count"] = len(content.splitlines())
        meta["complexity_score"] = self._complexity(content)

        return meta

    # ------------------------------------------------------------------

    def _base_type(self, type_str: str) -> str:
        """Strip wrapper types (&, mut, Option<>, Vec<>…) to get the leaf name."""
        t = re.sub(r"^&(?:mut\s+)?", "", type_str.strip())
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', t)
        return m.group(1) if m else t

    def _complexity(self, content: str) -> int:
        score = len(content.splitlines())
        depth = max_depth = 0
        for ch in content:
            if ch == "{":
                depth += 1
                if depth > max_depth:
                    max_depth = depth
            elif ch == "}":
                depth = max(depth - 1, 0)
        return score + max_depth * 5


# ---------------------------------------------------------------------------
# Python / CPython
# ---------------------------------------------------------------------------

class PythonMetadataCollector(BaseMetadataCollector):
    """
    Static + dynamic metadata for Python seeds.

    Dynamic collection:  a small probe snippet is appended to the seed
    before execution.  If the seed runs successfully the probe prints all
    top-level variable types as a JSON line starting with 'FFL_TYPES:'.
    This is the same key ('dynamic_types') already consumed by
    CPythonFusionStrategy for type-aware bridging.

    Liveness-aware bridging keys
    ----------------------------
    live_vars      list[str]       variables confirmed alive at program end
                                   (keys of dynamic_types — not deleted, not
                                    conditionally unset, not raised-over).
                                   Use THIS instead of syntactic top_level_vars
                                   when picking bridge sources from A.

    undefined_refs list[str]       bare identifiers that the seed USES at the
                                   top level but NEVER assigns, defines, or
                                   imports.  These are natural "holes" — ideal
                                   injection points for bridge values from A,
                                   because replacing them with 'fusion' fills a
                                   NameError rather than breaking internal logic.
    """

    language = "python"

    _ASSIGN_RE = re.compile(r'^([A-Za-z_]\w*)\s*=', re.MULTILINE)
    _IMPORT_RE = re.compile(r'^(?:import|from)\s+(\S+)', re.MULTILINE)
    _DEF_RE    = re.compile(r'^def\s+([A-Za-z_]\w*)\s*\(', re.MULTILINE)
    _CLASS_RE  = re.compile(r'^class\s+([A-Za-z_]\w*)', re.MULTILINE)
    # All bare identifier tokens (for undefined_refs computation)
    _TOKEN_RE  = re.compile(r'\b([A-Za-z_]\w*)\b')

    # Python builtins + keywords — always exclude from undefined_refs
    _BUILTINS: frozenset = frozenset({
        "None", "True", "False", "print", "len", "range", "type", "int", "str",
        "float", "list", "dict", "tuple", "set", "bool", "bytes", "bytearray",
        "object", "super", "isinstance", "issubclass", "hasattr", "getattr",
        "setattr", "delattr", "callable", "iter", "next", "enumerate", "zip",
        "map", "filter", "sorted", "reversed", "min", "max", "sum", "abs", "round",
        "pow", "divmod", "hex", "oct", "bin", "ord", "chr", "repr", "hash", "id",
        "input", "open", "vars", "dir", "help", "format", "staticmethod",
        "classmethod", "property", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "AttributeError", "RuntimeError", "StopIteration",
        "NotImplementedError", "OSError", "IOError", "ImportError", "NameError",
        "ZeroDivisionError", "OverflowError", "MemoryError", "AssertionError",
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "__name__", "__file__", "__doc__", "__package__", "__spec__",
        "__builtins__", "__import__",
    })

    # Appended to the seed during dry-run execution only.
    # Prints variable→type mapping as a JSON line without disturbing normal stdout.
    # Values are wrapped in lists to match CPythonFusionStrategy's consumption
    # pattern: `set(types1.get(va, []))`.
    _TYPE_PROBE = (
        "\nimport json as _ffl_json\n"
        "try:\n"
        "    _ffl_types = {\n"
        "        k: [type(v).__name__]\n"
        "        for k, v in list(globals().items())\n"
        "        if not k.startswith('_')\n"
        "        and not callable(v)\n"
        "        and not isinstance(v, type)\n"
        "    }\n"
        "    print('FFL_TYPES:' + _ffl_json.dumps(_ffl_types))\n"
        "except Exception:\n"
        "    pass\n"
    )

    @staticmethod
    def _strip_strings_and_comments(src: str) -> str:
        """
        Replace string literals and # comments with whitespace so that
        identifier tokens inside them are not counted as live references.
        Uses a simple state machine — fast and good enough for our purposes.
        """
        result = []
        i, n = 0, len(src)
        while i < n:
            ch = src[i]
            # Triple-quoted strings
            for q in ('"""', "'''"):
                if src[i:i+3] == q:
                    end = src.find(q, i + 3)
                    span = (end + 3) if end != -1 else n
                    result.append(" " * (span - i))
                    i = span
                    break
            else:
                # Single-line string
                if ch in ('"', "'"):
                    j = i + 1
                    while j < n and src[j] != ch and src[j] != "\n":
                        if src[j] == "\\":
                            j += 1
                        j += 1
                    span = j + 1
                    result.append(" " * (span - i))
                    i = span
                # Comment
                elif ch == "#":
                    j = src.find("\n", i)
                    span = j if j != -1 else n
                    result.append(" " * (span - i))
                    i = span
                else:
                    result.append(ch)
                    i += 1
        return "".join(result)

    def instrument_for_probe(self, content: str) -> str:
        return content + self._TYPE_PROBE

    def static_collect(self, content: str, filename: str = "") -> dict:
        meta: dict = {}

        # Top-level syntactic assignments (indentation 0)
        top_vars = list(dict.fromkeys(
            m.group(1) for m in self._ASSIGN_RE.finditer(content)
            if m.group(1) not in ("True", "False", "None")
        ))
        meta["top_level_vars"] = top_vars

        imports  = list(dict.fromkeys(self._IMPORT_RE.findall(content)))
        fns      = [m.group(1) for m in self._DEF_RE.finditer(content)]
        classes  = self._CLASS_RE.findall(content)
        meta["imports"]    = imports
        meta["functions"]  = fns
        meta["classes"]    = classes
        meta["line_count"] = len(content.splitlines())

        # ── Undefined references (ideal bridge injection points for B) ──────
        # These are bare identifiers the seed USES but never locally defines.
        # Replacing one with 'fusion' fills a dependency rather than breaking
        # an internal binding — it improves fusion validity instead of harming it.
        defined: set = (
            set(top_vars)
            | set(fns)
            | set(classes)
            | set(imports)
            | self._BUILTINS
        )
        # Collect all bare tokens that appear outside string/comment context.
        # We use a simple heuristic: strip string literals and comments first.
        stripped = self._strip_strings_and_comments(content)
        all_refs = set(self._TOKEN_RE.findall(stripped))
        meta["undefined_refs"] = sorted(all_refs - defined)

        return meta

    def dynamic_collect(self, content: str, result, filename: str = "") -> dict:
        """
        Parse 'FFL_TYPES:{...}' line from stdout.

        Returns:
          dynamic_types  dict[str, list[str]]  var → [type_name]
                                               (list format matches CPythonFusionStrategy)
          live_vars      list[str]             confirmed-alive var names
                                               (use for bridge source selection from A)
        """
        stdout = getattr(result, "stdout", "") or ""
        for line in stdout.splitlines():
            if line.startswith("FFL_TYPES:"):
                try:
                    raw = json.loads(line[len("FFL_TYPES:"):])
                    if not isinstance(raw, dict):
                        continue
                    # Normalise: ensure every value is a list of strings
                    dynamic_types = {}
                    for k, v in raw.items():
                        if isinstance(v, list):
                            dynamic_types[k] = v
                        else:
                            dynamic_types[k] = [str(v)]
                    return {
                        "dynamic_types": dynamic_types,
                        "live_vars":     sorted(dynamic_types.keys()),
                    }
                except Exception:
                    pass
        return {}


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

class GoMetadataCollector(BaseMetadataCollector):
    language = "go"

    _VAR_RE       = re.compile(r'\bvar\s+([a-zA-Z_]\w*)\s+([a-zA-Z_][\w.]*)')
    _SHORT_RE     = re.compile(r'\b([a-zA-Z_]\w*)\s*:=\s*(?:([a-zA-Z_][\w.]*)\s*\(|(\d+(?:\.\d+)?)|")')
    _STRUCT_RE    = re.compile(r'\btype\s+([a-zA-Z_]\w*)\s+struct')
    _GENERIC_RE   = re.compile(r'\[([A-Z]\w*)\s+(?:any|comparable|~)')

    def static_collect(self, content: str, filename: str = "") -> dict:
        meta: dict = {}

        var_types: Dict[str, str] = {}
        for m in self._VAR_RE.finditer(content):
            var_types[m.group(1)] = m.group(2)
        # Short declarations: infer type from literal or constructor
        for m in self._SHORT_RE.finditer(content):
            name = m.group(1)
            if name in var_types:
                continue
            if m.group(2):
                var_types[name] = m.group(2)       # e.g. int(5) → "int"
            elif m.group(3) and "." in m.group(3):
                var_types[name] = "float64"
            elif m.group(3):
                var_types[name] = "int"
            else:
                var_types[name] = "string"          # leading " → string
        meta["var_types"]     = var_types
        meta["struct_names"]  = self._STRUCT_RE.findall(content)
        meta["has_generics"]  = bool(self._GENERIC_RE.search(content))
        meta["line_count"]    = len(content.splitlines())

        return meta


# ---------------------------------------------------------------------------
# MLIR
# ---------------------------------------------------------------------------

class MLIRMetadataCollector(BaseMetadataCollector):
    language = "mlir"

    _FUNC_RE  = re.compile(r'func\.func\s+@([A-Za-z_][A-Za-z0-9_.$-]*)')
    _CONST_RE = re.compile(
        r'arith\.constant\s+(.*?)\s*:\s*([a-zA-Z0-9_<>{}\[\] ,:\?*\-]+)',
    )
    _TYPE_RE  = re.compile(
        r':\s*(i\d+|f\d+|index|memref<[^>]+>|vector<[^>]+>|tensor<[^>]+>)',
    )

    def static_collect(self, content: str, filename: str = "") -> dict:
        meta: dict = {}
        meta["functions"] = self._FUNC_RE.findall(content)
        constants = [
            {"value": m.group(1).strip(), "type": m.group(2).strip()}
            for m in self._CONST_RE.finditer(content)
        ]
        meta["constants"]   = constants
        meta["types_used"]  = list(dict.fromkeys(self._TYPE_RE.findall(content)))
        meta["line_count"]  = len(content.splitlines())
        return meta


# ---------------------------------------------------------------------------
# WGSL / naga / wgslc
# ---------------------------------------------------------------------------

class WGSLMetadataCollector(BaseMetadataCollector):
    language = "wgsl"

    _FN_RE     = re.compile(r'\bfn\s+([A-Za-z_]\w*)\s*\(')
    _STRUCT_RE = re.compile(r'\bstruct\s+([A-Za-z_]\w*)')
    _VAR_RE    = re.compile(
        r'\bvar\s*(?:<[^>]+>)?\s+([a-zA-Z_]\w*)\s*:\s*([A-Za-z_][\w<>,\s]*?)(?:\s*=|\s*;)',
    )

    def static_collect(self, content: str, filename: str = "") -> dict:
        from .fusion import NagaDeclarationFusionStrategy
        meta: dict = {}
        meta["functions"]  = self._FN_RE.findall(content)
        meta["structs"]    = self._STRUCT_RE.findall(content)
        var_types: Dict[str, str] = {}
        for m in self._VAR_RE.finditer(content):
            var_types[m.group(1)] = m.group(2).strip()
        meta["var_types"]  = var_types
        meta["line_count"] = len(content.splitlines())
        meta["has_declaration"] = NagaDeclarationFusionStrategy.has_injectable_declaration(content)
        return meta


# ---------------------------------------------------------------------------
# Clang / C / C++ / Obj-C
# ---------------------------------------------------------------------------

class ClangMetadataCollector(BaseMetadataCollector):
    """
    has_declaration caches whether the seed contains a struct/class/enum
    that core/fusion.py's ClangDeclarationFusionStrategy could donate (base
    class, template default, or item-nest source). Declaration fusion only
    produces a real (non-no-op) child when BOTH parents have one —
    ClangDeclarationFusionStrategy.is_viable_pair reads this flag so
    core/orchestrator.py can skip the technique for a pair instead of
    wasting a full compile on a fuse that's syntactically a no-op.
    """

    language = "clang"
    provides = ("has_declaration",)

    def static_collect(self, content: str, filename: str = "") -> dict:
        # Local import: avoids a module-load-time cycle (core.fusion doesn't
        # import core.dryrun, but importing at module scope here would still
        # force fusion.py — with its own heavier import chain — to load just
        # to run a --dry-run-only pass that doesn't need it otherwise).
        from .fusion import ClangDeclarationFusionStrategy
        return {
            "has_declaration": ClangDeclarationFusionStrategy.has_injectable_declaration(content),
        }


# ---------------------------------------------------------------------------
# Generic fallback
# ---------------------------------------------------------------------------

class GenericMetadataCollector(BaseMetadataCollector):
    language = "generic"

    def static_collect(self, content: str, filename: str = "") -> dict:
        return {
            "line_count": len(content.splitlines()),
            "byte_size":  len(content.encode("utf-8")),
        }


# ---------------------------------------------------------------------------
# JavaScript / V8
# ---------------------------------------------------------------------------

class JavaScriptMetadataCollector(BaseMetadataCollector):
    """Static facts about a JS seed.

    Unlike the Go and Rust collectors there is no type inference to do —
    JavaScript declarations carry no type — so this records what a *V8*
    seed is worth knowing for: which functions exist (declaration fusion
    needs a callable to point at), and how much of the engine's
    optimising, memory and internals surface the file touches.
    """

    language = "javascript"

    _FUNC_RE    = re.compile(r'\bfunction\s*\*?\s*([A-Za-z_$][\w$]*)\s*\(')
    _CLASS_RE   = re.compile(r'\bclass\s+([A-Za-z_$][\w$]*)')
    # let/const are the block-scoped forms that make a redeclaration a
    # SyntaxError, so they are the ones fusion has to track.
    _LEXICAL_RE = re.compile(r'^(?:let|const)\s+([A-Za-z_$][\w$]*)', re.M)
    # V8's %-prefixed runtime functions: the unsafe surface.
    _NATIVES_RE = re.compile(r'%([A-Z]\w*)\s*\(')
    _TYPED_RE   = re.compile(
        r'\b(?:ArrayBuffer|SharedArrayBuffer|DataView|Atomics|'
        r'(?:Ui|I)nt(?:8|16|32)Array|Float(?:32|64)Array|BigInt64Array)\b')

    def static_collect(self, content: str, filename: str = "") -> dict:
        natives = self._NATIVES_RE.findall(content)
        return {
            "function_names": self._FUNC_RE.findall(content),
            "class_names":    self._CLASS_RE.findall(content),
            "lexical_names":  self._LEXICAL_RE.findall(content),
            "natives":        sorted(set(natives)),
            "natives_count":  len(natives),
            "typed_array_count": len(self._TYPED_RE.findall(content)),
            "line_count":     len(content.splitlines()),
        }


# ---------------------------------------------------------------------------
# Collector registry
# ---------------------------------------------------------------------------

_COLLECTORS: Dict[str, BaseMetadataCollector] = {
    "rust":    RustMetadataCollector(),
    "python":  PythonMetadataCollector(),
    "cpython": PythonMetadataCollector(),
    "go":      GoMetadataCollector(),
    "javascript": JavaScriptMetadataCollector(),
    "js":      JavaScriptMetadataCollector(),
    "v8":      JavaScriptMetadataCollector(),
    "spidermonkey": JavaScriptMetadataCollector(),
    "mlir":    MLIRMetadataCollector(),
    "wgsl":    WGSLMetadataCollector(),
    "naga":    WGSLMetadataCollector(),
    "wgslc":   WGSLMetadataCollector(),
    # projects/clang/parser.py's ClangParser.parse_content sets "type" to
    # one of these three depending on extension (.c/.cpp,.cc,.cxx,.mm/.m).
    "c":       ClangMetadataCollector(),
    "cpp":     ClangMetadataCollector(),
    "objc":    ClangMetadataCollector(),
}
_generic_collector = GenericMetadataCollector()


def get_collector(language: str) -> BaseMetadataCollector:
    """Return the collector for *language*, falling back to the generic one."""
    return _COLLECTORS.get((language or "").lower(), _generic_collector)


# ---------------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------------

# One write-lock per DB path so concurrent threads don't trample each other.
_db_locks: Dict[str, threading.Lock] = {}
_db_locks_lock = threading.Lock()


def _get_db_lock(db_path: str) -> threading.Lock:
    with _db_locks_lock:
        if db_path not in _db_locks:
            _db_locks[db_path] = threading.Lock()
        return _db_locks[db_path]


def update_seed_metadata_in_db(db_path: str, identifier: str, new_meta: dict) -> None:
    """
    Merge *new_meta* into the stored JSON metadata for the seed identified
    by *identifier* and write it back.  Thread-safe.
    """
    if not db_path or not identifier:
        return
    lock = _get_db_lock(db_path)
    with lock:
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            row = conn.execute(
                "SELECT id, metadata FROM seeds WHERE identifier = ?",
                (identifier,),
            ).fetchone()
            if row is None:
                conn.close()
                return
            existing = json.loads(row[1]) if row[1] else {}
            for stale in OBSOLETE_METADATA_KEYS:
                existing.pop(stale, None)
            existing.update(new_meta)
            conn.execute(
                "UPDATE seeds SET metadata = ? WHERE id = ?",
                (json.dumps(existing), row[0]),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to update metadata for '{identifier}': {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _metadata_is_fresh(seed, fallback_language: Optional[str] = None) -> bool:
    """True when this seed's cached metadata already has every key the
    current --pre-analysis would write.

    Checking the keys rather than the pre_analysis_done flag is what lets a
    corpus analysed by an older build pick up fields added since — without
    it, a seed keeps its stale metadata for its entire life and every fusion
    recomputes from scratch what should have been cached once. Only the
    seeds that are actually missing something are re-executed.
    """
    meta = seed.metadata or {}
    if not meta.get("pre_analysis_done", False):
        return False
    if meta.get("pre_analysis_version", 0) != PRE_ANALYSIS_VERSION:
        return False       # analysed by a build whose output meant something else

    required = set()
    seed_type = str(meta.get("type") or "").lower()
    from .state_analysis import LANGUAGE_ALIASES
    if (LANGUAGE_ALIASES.get(seed_type)
            or LANGUAGE_ALIASES.get(str(fallback_language or "").lower())):
        # collect() writes these for any language state_analysis knows
        required.update(PRE_ANALYSIS_KEYS)
    required.update(get_collector(seed_type).provides)
    return required.issubset(meta)


def run_dryrun_with_metadata(
    seeds: list,
    driver_factory,
    db_path: Optional[str],
    concurrency: int = 4,
    timeout: int = 5,
    force: bool = False,
    collect_metadata: bool = True,
    filter_valid: bool = True,
    project_name: Optional[str] = None,
) -> list:
    """
    Execute seeds that still need it — at most once each, even if both
    concerns below are requested — then return the result.

    collect_metadata (--pre-analysis) and filter_valid (--dry-run) are
    independent: pass either alone, or both to reproduce this function's
    original combined behavior in one pass instead of two.

    Parameters
    ----------
    seeds            : list[Seed]  — corpus loaded from the project DB
    driver_factory   : callable()  — returns a fresh driver instance;
                                     called once per worker thread
    db_path          : str | None  — path to corpus.db for metadata
                                     updates; pass None to skip persistence
    concurrency      : int         — parallel worker count
    timeout          : int         — per-seed execution timeout (seconds).
                                     Bounds each individual seed's
                                     subprocess (BaseDriver._run_command
                                     kills the whole process group on
                                     expiry) — but see project_name below
                                     for why the pass can still stall.
    force            : bool        — if True, re-run even seeds that
                                     already satisfy the flags below
    collect_metadata : bool        — probe-instrument (where the language's
                                     collector needs one) and run static/
                                     dynamic/states-of-interest collection,
                                     persisting it and marking
                                     'pre_analysis_done'. Skipped seeds are
                                     ones that already have it, unless
                                     force. If False, the seed executes
                                     as-is with no collection at all.
    filter_valid     : bool        — drop rc != 0 seeds from the returned
                                     list. 'rc'/'dryrun_done' are always
                                     recorded from the execution regardless
                                     of this flag (it's free once the seed
                                     has run); this flag only controls
                                     whether the *return value* is filtered
                                     by it.
    project_name     : str | None  — when given, periodically runs core/
                                     driver.py's cleanup_stale_processes
                                     (same one core/orchestrator.py's main
                                     fuzzing loop calls every 2000
                                     iterations): the per-seed timeout only
                                     guarantees the host-side driver
                                     process is killed, not a Docker-
                                     contained or child process it spawned,
                                     so a run touching many seeds can
                                     accumulate leaked processes that
                                     starve later executions until the
                                     pass *looks* stuck even though each
                                     individual timeout still fired. None
                                     skips this (no project name to scope
                                     pkill patterns to).

    Returns
    -------
    list[Seed] — enriched seeds; rc==0 only if filter_valid, else all of
                 them (still enriched with whatever ran).
    """
    # Descriptive label for progress/log lines — reflects which pass(es)
    # this call is actually doing instead of always saying "dry-run".
    if collect_metadata and filter_valid:
        label = "dry-run+pre-analysis"
    elif collect_metadata:
        label = "pre-analysis"
    else:
        label = "dry-run"

    # Split into "already satisfies what this call needs" and "must (re)run".
    already_done: list = []
    to_run:       list = []

    for seed in seeds:
        meta = seed.metadata or {}
        needs_run = force
        if filter_valid and not meta.get("dryrun_done", False):
            needs_run = True
        if collect_metadata and not _metadata_is_fresh(seed, project_name):
            needs_run = True

        if needs_run:
            to_run.append(seed)
        elif not filter_valid or meta.get("rc", -1) == 0:
            already_done.append(seed)

    skipped = len(seeds) - len(to_run)
    if skipped:
        logger.info(
            f"  {label}: {skipped} seeds already satisfy the requested "
            f"pass(es) — skipping (pass force=True to re-run)"
        )

    if not to_run:
        logger.info(f"  {label}: nothing to execute, returning {len(already_done)} seeds.")
        return already_done

    logger.info(
        f"  {label}: executing {len(to_run)} seeds "
        f"(metadata={collect_metadata}, filter={filter_valid}, "
        f"timeout={timeout}s, workers={concurrency})"
    )

    _thread_local = threading.local()
    from_run: list = []
    done_count = 0
    total = len(to_run)
    _CLEANUP_EVERY = 500

    def _worker(seed):
        # Per-thread driver
        if not hasattr(_thread_local, "driver"):
            _thread_local.driver = driver_factory()
            _thread_local.driver.timeout = timeout
            # Tell drivers that support it to use minimal/stable flags
            if hasattr(_thread_local.driver, "dryrun_mode"):
                _thread_local.driver.dryrun_mode = True

        language  = (seed.metadata or {}).get("type", "unknown")
        collector = get_collector(language)

        # Only instrument with a runtime probe when metadata collection
        # actually needs one — a filter-only run should execute the
        # pristine seed, matching what "is this seed valid on its own"
        # should mean.
        if collect_metadata:
            probe_content = collector.instrument_for_probe(seed.content)
        else:
            probe_content = seed.content
        needs_probe = (probe_content != seed.content)

        # Create a temporary seed copy for execution so the original is unchanged
        from .fusion import Seed as _Seed
        run_seed = _Seed(
            id       = seed.id,
            content  = probe_content if needs_probe else seed.content,
            metadata = seed.metadata,
        )

        result = _thread_local.driver.execute(run_seed)

        # rc/dryrun_done are free once we've executed the seed at all.
        new_meta = {"dryrun_done": True, "rc": result.return_code}
        if collect_metadata:
            # Collect metadata using the *original* content for static
            # analysis but the execution result for dynamic analysis.
            new_meta.update(collector.collect(seed, result, fallback_language=project_name))
            new_meta["pre_analysis_done"] = True    # also set by collect(); kept for the
                                                    # collect_metadata-without-collector path

        return seed, result.return_code, new_meta

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, s): s for s in to_run}

        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                seed_out, rc, new_meta = fut.result()
            except Exception as e:
                logger.debug(f"{label} worker error for seed {seed.id}: {e}")
                done_count += 1
                continue

            # Enrich the seed in-memory
            if seed_out.metadata is None:
                seed_out.metadata = {}
            seed_out.metadata.update(new_meta)

            # Persist to DB (keyed by the 'filename' / identifier field)
            identifier = seed_out.metadata.get("filename", "")
            if identifier:
                update_seed_metadata_in_db(db_path, identifier, new_meta)

            if not filter_valid or rc == 0:
                from_run.append(seed_out)

            done_count += 1
            if project_name and done_count % _CLEANUP_EVERY == 0:
                # See project_name's docstring note above — bounds leaked
                # processes from accumulating across a long pass.
                from .driver import cleanup_stale_processes
                cleanup_stale_processes(project_name)
            if done_count % 500 == 0 or done_count == total:
                logger.info(f"  {label} progress: {done_count}/{total}")

    result_seeds = already_done + from_run
    logger.info(
        f"  {label} complete: {len(result_seeds)}/{len(seeds)} seeds returned"
        + (" (rc=0 only)" if filter_valid else " (unfiltered)")
    )
    return result_seeds
