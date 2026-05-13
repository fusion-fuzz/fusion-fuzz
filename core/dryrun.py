"""
core/dryrun.py — Dry-Run Metadata Collector for FusionFuzzLoop

During the dry-run phase (before fuzzing starts), every seed is executed
once. This module uses that opportunity to collect both *static*
(source-level regex analysis) and *dynamic* (stdout/stderr) metadata,
then persists it back to the corpus DB.

Why this matters for fusion quality
------------------------------------
The fusion strategies need to know what *types* the variables in a seed
carry so they can build valid bridge expressions (e.g. only bridge an
`i32` from seed A into a position that expects a numeric type in seed B).
Without this information the bridge is a random guess and will fail type
checking for strict languages like Rust, Swift, or Go.

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

All languages also receive:
  dryrun_done      bool            marker so a second run skips the seed
  rc               int             return code from the dry-run execution
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

class BaseMetadataCollector:
    """
    Collects static and optionally dynamic metadata from one seed execution.

    Subclasses override:
        static_collect(content, filename)  → dict
        dynamic_collect(content, result)   → dict   (optional)

    The result object has .return_code, .stdout, .stderr attributes.
    """

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

    def collect(self, seed, result) -> dict:
        """Combine static + dynamic metadata into one dict."""
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
        meta: dict = {}
        meta["functions"]  = self._FN_RE.findall(content)
        meta["structs"]    = self._STRUCT_RE.findall(content)
        var_types: Dict[str, str] = {}
        for m in self._VAR_RE.finditer(content):
            var_types[m.group(1)] = m.group(2).strip()
        meta["var_types"]  = var_types
        meta["line_count"] = len(content.splitlines())
        return meta


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
# Collector registry
# ---------------------------------------------------------------------------

_COLLECTORS: Dict[str, BaseMetadataCollector] = {
    "rust":    RustMetadataCollector(),
    "python":  PythonMetadataCollector(),
    "cpython": PythonMetadataCollector(),
    "go":      GoMetadataCollector(),
    "mlir":    MLIRMetadataCollector(),
    "wgsl":    WGSLMetadataCollector(),
    "naga":    WGSLMetadataCollector(),
    "wgslc":   WGSLMetadataCollector(),
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

def run_dryrun_with_metadata(
    seeds: list,
    driver_factory,
    db_path: Optional[str],
    concurrency: int = 4,
    timeout: int = 5,
    force: bool = False,
) -> list:
    """
    Execute every seed once, collect rich metadata, persist to *db_path*,
    and return only the seeds whose execution returned rc == 0.

    Parameters
    ----------
    seeds          : list[Seed]   — corpus loaded from the project DB
    driver_factory : callable()   — returns a fresh driver instance;
                                    called once per worker thread
    db_path        : str | None   — path to corpus.db for metadata updates;
                                    pass None to skip persistence
    concurrency    : int          — parallel worker count
    timeout        : int          — per-seed execution timeout (seconds)
    force          : bool         — if True, re-collect even for seeds that
                                    already have dryrun_done=True

    Returns
    -------
    valid_seeds : list[Seed]   — enriched seeds with rc == 0
    """
    # Split into "already processed" and "needs dry-run"
    already_valid: list = []
    to_run:        list = []

    for seed in seeds:
        done = (seed.metadata or {}).get("dryrun_done", False)
        if done and not force:
            if (seed.metadata or {}).get("rc", -1) == 0:
                already_valid.append(seed)
        else:
            to_run.append(seed)

    skipped = len(seeds) - len(to_run)
    if skipped:
        logger.info(
            f"  dry-run: {skipped} seeds already processed — skipping "
            f"(pass force=True to re-collect)"
        )

    if not to_run:
        logger.info(f"  dry-run: nothing to execute, returning {len(already_valid)} pre-validated seeds.")
        return already_valid

    logger.info(
        f"  dry-run: executing {len(to_run)} seeds "
        f"(timeout={timeout}s, workers={concurrency})"
    )

    _thread_local = threading.local()
    valid_from_run: list = []
    done_count = 0
    total = len(to_run)

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

        # Optionally instrument the seed with a runtime probe
        probe_content = collector.instrument_for_probe(seed.content)
        needs_probe   = (probe_content != seed.content)

        # Create a temporary seed copy for execution so the original is unchanged
        from .fusion import Seed as _Seed
        run_seed = _Seed(
            id       = seed.id,
            content  = probe_content if needs_probe else seed.content,
            metadata = seed.metadata,
        )

        result = _thread_local.driver.execute(run_seed)

        # Collect metadata using the *original* content for static analysis
        # but the execution result for dynamic analysis
        new_meta = collector.collect(seed, result)
        return seed, result.return_code, new_meta

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_worker, s): s for s in to_run}

        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                seed_out, rc, new_meta = fut.result()
            except Exception as e:
                logger.debug(f"dry-run worker error for seed {seed.id}: {e}")
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

            if rc == 0:
                valid_from_run.append(seed_out)

            done_count += 1
            if done_count % 500 == 0 or done_count == total:
                logger.info(f"  dry-run progress: {done_count}/{total}")

    valid_seeds = already_valid + valid_from_run
    logger.info(
        f"  dry-run complete: {len(valid_seeds)}/{len(seeds)} seeds are valid (rc=0)"
    )
    return valid_seeds
