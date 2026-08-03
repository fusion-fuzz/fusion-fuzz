import abc
import uuid
import random
import sqlite3
import os
import re
import io
import logging
import tokenize
import keyword
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from .mutation import BaseMutator, PHPMutator, CPythonMutator, RustMutator, HaskellMutator

logger = logging.getLogger("FFL.Fusion")

@dataclass
class Seed:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)

class FusionStrategy(abc.ABC):
    @abc.abstractmethod
    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        pass

    def is_viable_pair(self, parent_a: Seed, parent_b: Seed) -> bool:
        """True if this strategy is worth attempting on this specific parent
        pair. Default: always viable — most strategies degrade gracefully
        (e.g. dataflow fusion falls back to an on-the-fly scan) rather than
        needing a hard precondition. Override when a strategy can determine,
        from cached pre-analysis metadata, that it would just produce a
        syntactically-unchanged no-op fuse for this pair (see
        ClangDeclarationFusionStrategy) — core/orchestrator.py's
        process_iteration uses this to drop the strategy from the pair's
        technique pool instead of wasting a compile on a no-op."""
        return True

# ==========================================
# Helpers
# ==========================================

def pick_occurrence_count() -> int:
    """
    How many occurrences of a chosen bridge variable to replace in one
    dataflow-fusion substitution: usually a single clean edge into B, rarely
    two, almost never three — occasionally widening the bridge's reach
    without making broad cross-contamination the common case.
    """
    return random.choices((1, 2, 3), weights=(89, 10, 1))[0]

def replace_random_occurrence(s, old, new):
    """
    Simple string replacement for non-sensitive contexts. Replaces a
    weighted-random number of occurrences (see pick_occurrence_count)
    rather than always exactly one.
    """
    positions = []
    start = 0
    while True:
        start = s.find(old, start)
        if start == -1:
            break
        positions.append(start)
        start += len(old)

    if not positions:
        return s

    n = min(pick_occurrence_count(), len(positions))
    # Replace right-to-left so earlier offsets stay valid as we edit.
    for pos in sorted(random.sample(positions, n), reverse=True):
        s = s[:pos] + new + s[pos + len(old):]
    return s

def replace_random_occurrence_indented(s, old, new):
    """
    Context-aware replacement that respects indentation.
    Vital for Python to prevent IndentationError. Replaces a weighted-random
    number of occurrences (see pick_occurrence_count) rather than always
    exactly one.
    """
    matches = list(re.finditer(re.escape(old), s))
    if not matches:
        return s

    n = min(pick_occurrence_count(), len(matches))
    chosen = sorted(random.sample(matches, n), key=lambda m: m.start(), reverse=True)

    for match in chosen:
        start, end = match.span()

        line_start = s.rfind('\n', 0, start) + 1
        line_content = s[line_start:start]

        indent_match = re.match(r'^(\s*)', line_content)
        indent = indent_match.group(1) if indent_match else ""

        if '\n' in new:
            lines = new.split('\n')
            indented_new = lines[0] + '\n' + '\n'.join([indent + line for line in lines[1:]])
            replacement = indented_new
        else:
            replacement = new

        s = s[:start] + replacement + s[end:]
    return s

def replace_word_occurrences(s: str, old: str, new: str, ignorecase: bool = False) -> str:
    """
    Word-boundary-safe replacement — unlike PHP's `$`-sigil or MLIR's
    `%`-sigil variables, bare identifiers (C, Swift, Fortran) have no
    unambiguous prefix, so a naive substring replace can corrupt unrelated
    identifiers or keywords that merely contain `old` as a substring (e.g.
    replacing bare `r` would turn `struct` into `styuct`, or `for` into
    `foy`). `ignorecase` is for Fortran, which is case-insensitive ('X' and
    'x' are the same identifier). Replaces a weighted-random number of
    occurrences (see pick_occurrence_count) rather than always exactly one.
    """
    flags = re.IGNORECASE if ignorecase else 0
    matches = list(re.finditer(rf'(?<!\w){re.escape(old)}(?!\w)', s, flags))
    if not matches:
        return s
    n = min(pick_occurrence_count(), len(matches))
    chosen = sorted(random.sample(matches, n), key=lambda m: m.start(), reverse=True)
    for match in chosen:
        start, end = match.span()
        s = s[:start] + new + s[end:]
    return s

# ==========================================
# Python Fusion Helpers
# ==========================================

KW = set(keyword.kwlist)
ASSIGN_RE = re.compile(r'^([A-Za-z_]\w*)\s*=', re.ASCII)
FUTURE_RE = re.compile(r'^\s*from\s+__future__\s+import\b')
TOKEN_RE  = re.compile(r'\b([A-Za-z_]\w*)\b', re.ASCII)

def strip_and_collect_future(src: str):
    futures, keep = [], []
    for ln in src.splitlines(keepends=True):
        if FUTURE_RE.match(ln):
            futures.append(ln.strip())
        else:
            keep.append(ln)
    return keep, sorted(set(futures))

def collect_top_level_assigned_vars(src_text):
    """
    Returns a list of all variables assigned at the top level (indentation 0).
    """
    cands = []
    for ln in src_text.splitlines():
        if ln.startswith((" ", "\t")): continue
        m = ASSIGN_RE.match(ln)
        if m:
            name = m.group(1)
            if name not in KW:
                cands.append(name)
    return cands

def pick_top_level_assigned_var(src_text):
    cands = collect_top_level_assigned_vars(src_text)
    if not cands: return None
    return cands[-1]

def collect_bare_vars(src_text):
    out = []
    for m in TOKEN_RE.finditer(src_text):
        name = m.group(1)
        if name in KW: continue
        start, end = m.span()
        prev_ch = src_text[start-1] if start > 0 else ""
        next_ch = src_text[end] if end < len(src_text) else ""
        if prev_ch == "." or re.match(r'\s*\.', next_ch): continue
        out.append(name)
    return out

def replace_one_b_occurrence(src_text, name, replacement):
    """
    Replaces a weighted-random number of occurrences (see
    pick_occurrence_count) of a variable in B with 'replacement', handling
    indentation context for each.
    """
    if not name: return src_text, False

    patt = re.compile(rf'(?<!\.)\b{re.escape(name)}\b(?!\s*\.)')

    matches = list(patt.finditer(src_text))
    if not matches:
        return src_text, False

    n = min(pick_occurrence_count(), len(matches))
    chosen = sorted(random.sample(matches, n), key=lambda m: m.start(), reverse=True)

    for match in chosen:
        start, end = match.span()

        line_start = src_text.rfind('\n', 0, start) + 1
        indent_match = re.match(r'^(\s*)', src_text[line_start:start])
        indent = indent_match.group(1) if indent_match else ""

        final_repl = replacement
        if '\n' in replacement:
            lines = replacement.split('\n')
            final_repl = lines[0] + '\n' + '\n'.join([indent + l for l in lines[1:]])

        src_text = src_text[:start] + final_repl + src_text[end:]

    return src_text, True

def mutate_constants_and_rhs_ops(src_text: str):
    # This logic is now handled by CPythonMutator in mutation.py
    return src_text, {} 

# ==========================================
# MLIR Specific Helpers
# ==========================================

MLIR_FUNC_DEF = re.compile(r'(func\.func\s+@)([A-Za-z_0-9_.$-]+)')
# Upgrade bare `func @name` (LLVM <23 syntax) to `func.func @name` in place.
# Negative lookbehind on both word chars and `.` avoids double-patching
# already-correct `func.func @` occurrences.
_MLIR_BARE_FUNC_RE = re.compile(r'(?<![\w.])func\s+@')
MLIR_GLOB_DEF = re.compile(r'(memref\.global\s+@)([A-Za-z_0-9_.$-]+)')
MLIR_CONST_RE = re.compile(
    r'^\s*(%[A-Za-z0-9_.$-]+)\s*=\s*arith\.constant\s+(.*?)\s*:\s*([A-Za-z0-9_<>{}\[\], :\?*\-]+)\s*$',
    re.M
)

def mlir_rename_symbols(src: str, prefix: str):
    defs = []
    def repl_func(m):
        defs.append(m.group(2))
        return m.group(1) + prefix + m.group(2)
    out = MLIR_FUNC_DEF.sub(repl_func, src)
    def repl_glob(m):
        defs.append(m.group(2))
        return m.group(1) + prefix + m.group(2)
    out = MLIR_GLOB_DEF.sub(repl_glob, out)
    for name in sorted(set(defs), key=len, reverse=True):
        out = re.sub(r'@' + re.escape(name) + r'\b', '@' + prefix + name, out)
    return out

def mlir_extract_constants(src: str):
    out = []
    for m in MLIR_CONST_RE.finditer(src):
        res, lit, ty = m.group(1), m.group(2), m.group(3).strip()
        out.append({"res": res, "lit": lit, "ty": ty, "span": m.span(), "line": m.group(0)})
    return out

def mlir_strip_directives(src: str) -> str:
    """
    Remove FileCheck / lit test-runner directive lines from MLIR source.
    These are comment lines meaningful only to the test harness, not to
    mlir-opt.  Keeping them causes two problems:
      1. mlir_strip_outer_module sees 'module' at a non-zero offset and
         returns the source unchanged, producing illegal nested modules.
      2. --verify-diagnostics mode expects 'expected-error' annotations
         that no longer match after fusion, causing spurious failures.
    """
    _DIR_RE = re.compile(
        r'^\s*//\s*(?:RUN:|CHECK(?:-[A-Z]+)?:|XFAIL:|REQUIRES:|UNSUPPORTED:|'
        r'expected-(?:error|warning|note|remark))',
        re.IGNORECASE,
    )
    lines = [ln for ln in src.splitlines() if not _DIR_RE.match(ln)]
    return "\n".join(lines)

def mlir_strip_outer_module(src: str) -> str:
    """
    Strip the outermost 'module { ... }' wrapper.
    Skips leading blank lines, // comment lines, and file-level attribute/type
    alias lines (#alias = ..., !alias = ...) so that seeds with preambles are
    handled correctly.
    Returns src unchanged when no top-level module wrapper is found.
    """
    s = src
    # Walk past leading whitespace, // comment lines, and #/! alias definitions.
    while True:
        s = s.lstrip()
        if s.startswith("//"):
            nl = s.find("\n")
            s = s[nl + 1:] if nl != -1 else ""
        elif s.startswith("#") or s.startswith("!"):
            # Skip attribute/type alias definition lines (#alias = ..., !alias = ...)
            nl = s.find("\n")
            s = s[nl + 1:] if nl != -1 else ""
        else:
            break
    if not s.startswith("module"):
        return src
    lb = s.find("{")
    if lb == -1:
        return src
    depth = 0
    end = -1
    for i, ch in enumerate(s[lb:], start=lb):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end == -1:
        return src
    return s[lb + 1:end]

# ==========================================
# Generic Strategies
# ==========================================

class GenericDataflowStrategy(FusionStrategy):
    def __init__(self, mutator: BaseMutator = None, lightweight: bool = False):
        self.mut = mutator if mutator else BaseMutator()
        self.bridge_var_name = "fusion"
        # --pre-analysis off: bypass whatever parse-time dataflow-graph
        # metadata (dataflow1/dataflow2 below) this strategy would
        # otherwise consume and do a fresh, on-the-fly variable scan +
        # single random connect instead — see _lightweight_bridge.
        self.lightweight = lightweight

    @abc.abstractmethod
    def format_assignment(self, lhs: str, rhs: str) -> str:
        pass

    # --- lightweight (--pre-analysis off) fallback ----------------------

    _LIGHTWEIGHT_VAR_RE = re.compile(r'\b[A-Za-z_]\w*\b')

    def _lightweight_vars(self, code: str) -> List[str]:
        """On-the-fly variable-name scan used in place of dataflow1/
        dataflow2 when --pre-analysis is off. Subclasses override for
        their own variable syntax (PHP's $var sigil, Fortran's
        declarations, ...)."""
        return self._LIGHTWEIGHT_VAR_RE.findall(code)

    def _lightweight_replace(self, code: str, var: str, bridge: str) -> str:
        """Substitute a weighted-random number of occurrences of `var`
        with `bridge` in `code`. Subclasses override for identifier
        syntax that needs word-boundary safety (bare C/Fortran
        identifiers, where a substring replace could corrupt an unrelated
        identifier)."""
        return replace_random_occurrence(code, var, bridge)

    def _lightweight_bridge(self, code1: str, code2: str):
        """Very lightweight dataflow fusion for when --pre-analysis hasn't
        populated dataflow1/dataflow2: scan each side's variables on the
        fly (no dependency graph, no co-occurrence grouping) and connect
        one uniformly random pair."""
        vars1 = self._lightweight_vars(code1)
        vars2 = self._lightweight_vars(code2)
        if not vars1 or not vars2:
            return code1, code2
        var1 = random.choice(vars1)
        var2 = random.choice(vars2)
        bridge = self.bridge_var_name
        code1 = code1 + f"\n{self.format_assignment(bridge, var1)}\n"
        code2 = self._lightweight_replace(code2, var2, bridge)
        return code1, code2

    def interleave_code_blocks(self, code1, code2, dataflow1, dataflow2, extra_flows=None):
        if self.lightweight:
            return self._lightweight_bridge(code1, code2)
        if not dataflow1 or not dataflow2:
            return code1, code2

        if extra_flows:
            dataflow1 = dataflow1 + extra_flows

        bridge = self.bridge_var_name

        if random.choice([True, False]):
            try:
                group1 = random.choice(dataflow1)
                var1 = random.choice(group1)
                group2 = random.choice(dataflow2)
                var2 = random.choice(group2)
                bridge_stmt = self.format_assignment(bridge, var1)
                code1 += f"\n{bridge_stmt}\n"
                
                # Use context-aware replacement if Python-like indentation matters
                # For Generic, we assume loose (like PHP), but we can check subclass
                if isinstance(self, CPythonFusionStrategy):
                    code2 = replace_random_occurrence_indented(code2, var2, bridge)
                else:
                    code2 = replace_random_occurrence(code2, var2, bridge)
            except IndexError:
                pass
            return code1, code2

        max_df1 = max(dataflow1, key=len) if dataflow1 else []
        max_df2 = max(dataflow2, key=len) if dataflow2 else []

        if max_df1 and max_df2:
            var1 = random.choice(max_df1)
            var2 = random.choice(max_df2)
            bridge_stmt = self.format_assignment(bridge, var1)
            code1 += f"\n{bridge_stmt}\n"
            
            if isinstance(self, CPythonFusionStrategy):
                code2 = replace_random_occurrence_indented(code2, var2, bridge)
            else:
                code2 = replace_random_occurrence(code2, var2, bridge)

        return code1, code2

# ==========================================
# PHP Specific Fusion Strategy
# ==========================================

class PHPFusionStrategy(GenericDataflowStrategy):
    # PHP variables carry a $ sigil, distinctive enough for the base
    # class's plain substring _lightweight_replace to stay safe.
    _LIGHTWEIGHT_VAR_RE = re.compile(r'\$[A-Za-z_]\w*')

    def __init__(self, project_root="projects/php", lightweight: bool = False):
        super().__init__(mutator=PHPMutator(), lightweight=lightweight)
        self.project_root = project_root
        self.bridge_var_name = "$fusion"
        self.apifuzz = True
        self.ini = True
        self.mutation = True
        self.stmt_fusion = False
        self.dataflow_fusion = True
        self.all_fusion = False
        self.apis = []
        self.classes = []
        self._load_apis()
        self._load_classes()

    def format_assignment(self, lhs: str, rhs: str) -> str:
        return f"{lhs} = {rhs};"

    def _load_apis(self):
        db_path = os.path.join(self.project_root, "apis.db")
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name, num_params FROM functions")
            self.apis = cursor.fetchall()
            conn.close()
        except Exception as e:
            print(f"Error loading APIs: {e}")

    def _load_classes(self):
        db_path = os.path.join(self.project_root, "class.db")
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT id, class_name FROM classes")
            class_rows = cursor.fetchall()
            self.classes = []
            for cls in class_rows:
                cls_id = cls[0]
                cls_name = cls[1]
                cursor.execute("SELECT name FROM attributes WHERE class_id = ?", (cls_id,))
                attrs = [r[0] for r in cursor.fetchall()]
                cursor.execute("SELECT name, params_count FROM methods WHERE class_id = ?", (cls_id,))
                methods = [{'name': r[0], 'params_count': r[1]} for r in cursor.fetchall()]
                self.classes.append({'name': cls_name, 'attributes': attrs, 'methods': methods})
            conn.close()
        except Exception as e:
            print(f"Error loading Classes: {e}")

    def random_jit_mode(self):
        jit_mode = random.choice(['1254','1205'])
        return f"\nopcache.enable=1\nopcache.enable_cli=1\nopcache.jit={jit_mode}\n"

    def get_random_config(self):
        config_options = {
            "precision": random.choice([10, 12, 13, 14, 17]),
            "serialize_precision": random.choice([5, 10, 14, 15, 75, -1]),
            "memory_limit": random.choice(["100M", "256M", "512M", "128M", "6G", "-1"]),
            "max_execution_time": random.choice([0, 1, 2, 10, 12, 60]),
            "opcache.enable": random.choice([0, 1]),
            "opcache.enable_cli": random.choice([0, 1]),
            "opcache.jit": random.choice([0, 1205, 1235, 1255]),
            "error_reporting": random.choice([0, -1, "E_ALL"]),
        }
        random_key = random.choice(list(config_options.keys()))
        return f"{random_key}={config_options[random_key]}"

    def random_inis(self):
        if not self.ini: return ""
        inis = self.get_random_config() + '\n'
        if random.choice([True, False, False, False]):
            inis += self.random_jit_mode()
        return inis

    # Regex helpers for PHP symbol extraction
    _PHP_IDENT = r'[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*'
    _PHP_FUNC_DEF_RE = re.compile(r'(?<=function )([a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)')
    _PHP_CLASS_DEF_RE = re.compile(r'(?:(?<=class )|(?<=interface )|(?<=trait )|(?<=enum ))([a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)')
    _PHP_PREAMBLE_RE = re.compile(r'^\s*(?:declare\s*\(|namespace\s+)', re.M)
    _PHP_SKIP_NAMES = frozenset({
        '__construct', '__destruct', '__toString', '__invoke', '__clone',
        '__get', '__set', '__isset', '__unset', '__call', '__callStatic',
        '__sleep', '__wakeup', '__serialize', '__unserialize', '__debugInfo',
        'get', 'set',
    })

    def _extract_preamble(self, code: str):
        """Split code into (preamble_lines, rest) where preamble contains declare/namespace."""
        preamble, rest = [], []
        for line in code.splitlines():
            if self._PHP_PREAMBLE_RE.match(line):
                preamble.append(line.strip())
            else:
                rest.append(line)
        return preamble, '\n'.join(rest)

    def _extract_top_level_names(self, code: str):
        return (set(self._PHP_FUNC_DEF_RE.findall(code)) |
                set(self._PHP_CLASS_DEF_RE.findall(code))) - self._PHP_SKIP_NAMES

    def _resolve_name_conflicts(self, code_a: str, code_b: str) -> str:
        """Rename functions/classes in code_b that clash with names in code_a."""
        conflicts = self._extract_top_level_names(code_a) & self._extract_top_level_names(code_b)
        if not conflicts:
            return code_b
        result = code_b
        for name in sorted(conflicts, key=len, reverse=True):
            # Replace word-boundary occurrences not preceded by -> or ::
            result = re.sub(
                rf'(?<![->.:\'"` ])\b{re.escape(name)}\b',
                name + '_ffl',
                result,
            )
        return result

    def _instrumentation_classfuzz(self, defined_vars) -> Tuple[str, str, List[str]]:
        if not self.classes: return "", "", []
        _after_instrument = []
        new_vars = []
        try:
            class_info = random.choice(self.classes)
            class_name = class_info['name']
            # Wrap constructor in try/catch — many classes require constructor args.
            pre_str = (
                f"\ntry {{ $cls = new {class_name}(); }}"
                f" catch (\\Throwable $_e) {{ $cls = new stdClass(); }}\n"
            )
            new_vars.append("$cls")
            if class_info['attributes']:
                attr_name = random.choice(class_info['attributes'])
                pre_str += f"try {{ $clsAttr=$cls->{attr_name}; }} catch (\\Throwable $_e) {{}}\n"
                new_vars.append("$clsAttr")
            if class_info['methods']:
                method_info = random.choice(class_info['methods'])
                method_name = method_info['name']
                params_count = method_info['params_count']
                vars_pool = defined_vars if defined_vars else ["'test'", "0"]
                for _ in range(5):
                    args = [random.choice(vars_pool) for _ in range(params_count)]
                    _call = f"$cls->{method_name}({','.join(args)});"
                    _wrapper = f"try {{ {_call} }} catch (\\Throwable $e) {{}};"
                    _after_instrument.append(_wrapper)
                _after_str = '\n'.join(_after_instrument) + '\n'
            else:
                _after_str = ""
            return pre_str, _after_str, new_vars
        except Exception:
            return "", "", []

    def select_random_function(self):
        if not self.apis: return "var_dump", 1
        return random.choice(self.apis)

    def _instrumentation_apifuzz(self, defined_vars):
        if not self.apis: return ""
        _instruments = []
        func, param_num = self.select_random_function()
        vars_pool = defined_vars if defined_vars else ["'test'", "0", "null"]
        for _ in range(5):
            args = []
            for _ in range(param_num):
                args.append(random.choice(vars_pool))
            _call = f"{func}({','.join(args)});"
            _wrapper = f"try {{ {_call} }} catch (\\Throwable $e) {{}};"
            _instruments.append(_wrapper)
        return '\n'.join(_instruments) + '\n'

    # ── Statement Fusion helpers ──────────────────────────────────

    _PHP_VAR_RE = re.compile(r'\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*')
    _PHP_FUNC_CALL_RE = re.compile(r'(?<![->:])\b([a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)\s*\(')
    _PHP_NEW_RE = re.compile(r'\bnew\s+([a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)')
    _PHP_ASSIGN_RE = re.compile(r'(\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*)\s*=[^=]')
    _PHP_COMPOUND_KW = re.compile(
        r'^\s*(?:function\s|class\s|interface\s|trait\s|enum\s|abstract\s+class\s'
        r'|if\s*\(|else\s*\{|elseif\s*\(|else\s+if\s*\('
        r'|for\s*\(|foreach\s*\(|while\s*\(|do\s*\{'
        r'|switch\s*\(|match\s*\('
        r'|try\s*\{|catch\s*\(|finally\s*\{)')
    _PHP_NORM_VAR = re.compile(r'\$[a-zA-Z_\x80-\xff][a-zA-Z0-9_\x80-\xff]*')
    _PHP_NORM_STR = re.compile(r"(?:\"[^\"]*\"|'[^']*')")
    _PHP_NORM_NUM = re.compile(r'\b\d+(?:\.\d+)?\b')

    _PHP_CONTINUATION_RE = re.compile(
        r'^\s*(?:catch\s*\(|finally\s*\{|else\s*\{|elseif\s*\(|else\s+if\s*\()')

    @classmethod
    def _split_statements(cls, code: str) -> List[str]:
        """Split PHP code into statement units respecting brace/paren depth
        and string literals.  Compound blocks (functions, classes, control
        structures including try/catch/finally and if/elseif/else chains)
        are kept as single units."""
        statements: List[str] = []
        current: List[str] = []
        brace_depth = 0
        paren_depth = 0
        in_sq = False
        in_dq = False
        in_heredoc = False
        heredoc_tag = ""
        escaped = False
        i = 0
        n = len(code)

        def _lookahead_is_continuation(pos: int) -> bool:
            """Check if text after pos starts with catch/finally/else/elseif."""
            rest = code[pos:]
            return bool(cls._PHP_CONTINUATION_RE.match(rest))

        while i < n:
            ch = code[i]
            nch = code[i + 1] if i + 1 < n else ''

            # ── string literal tracking ──
            if escaped:
                current.append(ch)
                escaped = False
                i += 1
                continue
            if ch == '\\' and (in_sq or in_dq):
                current.append(ch)
                escaped = True
                i += 1
                continue
            if in_sq:
                current.append(ch)
                if ch == "'":
                    in_sq = False
                i += 1
                continue
            if in_dq:
                current.append(ch)
                if ch == '"':
                    in_dq = False
                i += 1
                continue
            if in_heredoc:
                current.append(ch)
                if ch == '\n':
                    rest = code[i + 1:]
                    if rest.startswith(heredoc_tag + ';') or rest.startswith(heredoc_tag + '\n') or rest.rstrip() == heredoc_tag:
                        end_len = len(heredoc_tag)
                        current.append(code[i + 1:i + 1 + end_len])
                        i += 1 + end_len
                        in_heredoc = False
                        if i < n and code[i] == ';':
                            current.append(';')
                            i += 1
                        continue
                i += 1
                continue

            # ── skip single-line comments ──
            if ch == '/' and nch == '/':
                while i < n and code[i] != '\n':
                    current.append(code[i])
                    i += 1
                continue
            if ch == '#' and nch != '[':
                while i < n and code[i] != '\n':
                    current.append(code[i])
                    i += 1
                continue
            # ── skip multi-line comments ──
            if ch == '/' and nch == '*':
                current.append(ch)
                i += 1
                while i < n:
                    current.append(code[i])
                    if code[i] == '*' and i + 1 < n and code[i + 1] == '/':
                        current.append('/')
                        i += 2
                        break
                    i += 1
                continue

            # ── detect string starts ──
            if ch == "'" and brace_depth + paren_depth >= 0:
                in_sq = True
                current.append(ch)
                i += 1
                continue
            if ch == '"' and brace_depth + paren_depth >= 0:
                in_dq = True
                current.append(ch)
                i += 1
                continue
            # ── heredoc / nowdoc ──
            if ch == '<' and code[i:i+3] == '<<<':
                current.append('<<<')
                i += 3
                tag_start = i
                while i < n and code[i] not in ('\n', '\r'):
                    i += 1
                raw_tag = code[tag_start:i].strip().strip("'\"")
                heredoc_tag = raw_tag
                current.append(code[tag_start:i])
                in_heredoc = True
                continue

            current.append(ch)

            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth <= 0 and paren_depth <= 0:
                    brace_depth = 0
                    # Before emitting, check if the next non-whitespace is a
                    # continuation keyword (catch, finally, else, elseif).
                    # If so, keep accumulating into the same statement.
                    j = i + 1
                    while j < n and code[j] in (' ', '\t', '\n', '\r'):
                        j += 1
                    if _lookahead_is_continuation(j):
                        # Absorb whitespace and continue
                        i += 1
                        continue
                    stmt = ''.join(current).strip()
                    if stmt:
                        statements.append(stmt)
                    current = []
                    i += 1
                    continue
            elif ch == ';' and brace_depth <= 0 and paren_depth <= 0:
                stmt = ''.join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
                i += 1
                continue

            i += 1

        leftover = ''.join(current).strip()
        if leftover:
            statements.append(leftover)
        return statements

    def _stmt_defines_uses(self, stmt: str):
        """Return (defines: set, uses: set) for a single PHP statement.
        'defines' = variable assignments + function/class declarations.
        'uses'    = variables and user-defined function/class references
                    that appear but are not the assignment target."""
        defines = set()
        uses = set()

        # Function / class / interface / trait / enum definitions
        for name in self._PHP_FUNC_DEF_RE.findall(stmt):
            if name not in self._PHP_SKIP_NAMES:
                defines.add(name)
        for name in self._PHP_CLASS_DEF_RE.findall(stmt):
            if name not in self._PHP_SKIP_NAMES:
                defines.add(name)

        # Variable assignments: $x = ...
        for m in self._PHP_ASSIGN_RE.finditer(stmt):
            defines.add(m.group(1))

        # All variables referenced
        all_vars = set(self._PHP_VAR_RE.findall(stmt))
        uses |= all_vars

        # Function calls (user-defined, not builtins — we can't distinguish
        # perfectly, but that's fine: extra edges just add ordering constraints)
        for m in self._PHP_FUNC_CALL_RE.finditer(stmt):
            uses.add(m.group(1))

        # new ClassName
        for m in self._PHP_NEW_RE.finditer(stmt):
            uses.add(m.group(1))

        return defines, uses

    def _normalize_stmt(self, stmt: str) -> str:
        """Normalize a statement for similarity comparison:
        strip variable names, string literals, and numeric literals."""
        s = self._PHP_NORM_VAR.sub('$_', stmt)
        s = self._PHP_NORM_STR.sub('"_"', s)
        s = self._PHP_NORM_NUM.sub('0', s)
        return s

    @staticmethod
    def _token_similarity(norm_a: str, norm_b: str) -> float:
        """Jaccard similarity on whitespace-split tokens of normalized statements."""
        toks_a = set(norm_a.split())
        toks_b = set(norm_b.split())
        if not toks_a and not toks_b:
            return 1.0
        intersection = toks_a & toks_b
        union = toks_a | toks_b
        return len(intersection) / len(union) if union else 0.0

    def _dependency_graph_interleave(self, stmts_a: List[str], stmts_b: List[str]) -> List[str]:
        """Interleave statements from seed A and seed B using dependency-graph
        topological sort with token-similarity tie-breaking.

        1. Build a dependency DAG across all statements (A ∪ B).
        2. Topologically sort: at each step, among ready statements (all
           dependencies met), pick the one most similar to the last emitted
           statement (with randomness in the top-k for diversity).
        3. The result is a valid interleaving that respects def-use order
           and clusters structurally similar statements.
        """
        # Tag each statement with its origin for bookkeeping
        tagged = [(s, 'a') for s in stmts_a] + [(s, 'b') for s in stmts_b]
        n = len(tagged)
        if n == 0:
            return []

        # Pre-compute defines/uses and normalized forms
        info = []
        for stmt, origin in tagged:
            defines, uses = self._stmt_defines_uses(stmt)
            norm = self._normalize_stmt(stmt)
            info.append({
                'stmt': stmt,
                'origin': origin,
                'defines': defines,
                'uses': uses,
                'norm': norm,
            })

        # Build dependency edges: stmt j depends on stmt i if j uses
        # something i defines.  We only add the edge to the *last* definer
        # within the same seed to avoid over-constraining across seeds
        # (cross-seed deps don't exist in the original programs).
        # However, for function/class names we add cross-seed edges too,
        # since a call to a function defined in the other seed must come after.
        deps = [set() for _ in range(n)]  # deps[j] = set of indices j depends on

        # Map: name → list of (index, origin) that define it
        def_map: Dict[str, List[Tuple[int, str]]] = {}
        for i, inf in enumerate(info):
            for name in inf['defines']:
                def_map.setdefault(name, []).append((i, inf['origin']))

        for j, inf_j in enumerate(info):
            for name in inf_j['uses']:
                if name not in def_map:
                    continue
                definers = def_map[name]
                for di, d_origin in definers:
                    if di == j:
                        continue
                    # Same-seed edge: always add (preserves original order intent)
                    if d_origin == inf_j['origin']:
                        deps[j].add(di)
                    else:
                        # Cross-seed edge: only for function/class definitions
                        # (not variables — cross-seed variable refs are intentionally
                        # invalid to stress the interpreter)
                        if not name.startswith('$'):
                            deps[j].add(di)

        # Topological sort with similarity tie-breaking
        emitted = [False] * n
        emit_count = [0]  # use list for mutability in nested func
        result: List[str] = []

        # in-degree for each node
        in_degree = [len(d) for d in deps]

        # reverse map: who depends on me
        dependents = [[] for _ in range(n)]
        for j in range(n):
            for di in deps[j]:
                dependents[di].append(j)

        ready = [i for i in range(n) if in_degree[i] == 0]

        last_norm = ""
        while ready:
            if not result:
                pick_idx = random.choice(ready)
            else:
                # Score each ready statement by similarity to the last emitted
                scored = []
                for ri in ready:
                    sim = self._token_similarity(last_norm, info[ri]['norm'])
                    scored.append((sim, ri))
                scored.sort(key=lambda x: -x[0])
                # Pick from top-3 for diversity
                top_k = min(3, len(scored))
                pick_idx = random.choice([s[1] for s in scored[:top_k]])

            ready.remove(pick_idx)
            emitted[pick_idx] = True
            result.append(info[pick_idx]['stmt'])
            last_norm = info[pick_idx]['norm']

            # Unblock dependents
            for dep_j in dependents[pick_idx]:
                in_degree[dep_j] -= 1
                if in_degree[dep_j] == 0 and not emitted[dep_j]:
                    ready.append(dep_j)

        # If there are remaining statements (cycles — shouldn't happen with
        # valid seeds, but be defensive), append them in original order
        for i in range(n):
            if not emitted[i]:
                result.append(info[i]['stmt'])

        return result

    def _stmt_cross_replace_variable(self, code: str, vars_a: List[str], vars_b: List[str]) -> str:
        """Pick one random variable from B's set and replace one random
        occurrence with a random variable from A's set."""
        if not vars_a or not vars_b:
            return code
        var_b = random.choice(vars_b)
        var_a = random.choice(vars_a)
        if var_a == var_b:
            return code
        return replace_random_occurrence(code, var_b, var_a)

    _PHP_BLOCK_HEAD_RE = re.compile(
        r'^\s*(?:function\s|class\s|interface\s|trait\s|enum\s|abstract\s+class\s'
        r'|if\s*\(|elseif\s*\(|else\s*\{'
        r'|for\s*\(|foreach\s*\(|while\s*\(|do\s*\{'
        r'|switch\s*\(|match\s*\('
        r'|try\s*\{|catch\s*\(|finally\s*\{)')

    @staticmethod
    def _find_outermost_brace_body(stmt: str):
        """Find the span of the outermost { body } in a statement.
        Returns (body_start, body_end) indices into stmt where body_start
        is the index after '{' and body_end is the index of the matching '}'.
        Returns None if no brace block found."""
        in_sq = False
        in_dq = False
        escaped = False
        depth = 0
        body_start = -1
        for i, ch in enumerate(stmt):
            if escaped:
                escaped = False
                continue
            if ch == '\\' and (in_sq or in_dq):
                escaped = True
                continue
            if in_sq:
                if ch == "'":
                    in_sq = False
                continue
            if in_dq:
                if ch == '"':
                    in_dq = False
                continue
            if ch == "'":
                in_sq = True
                continue
            if ch == '"':
                in_dq = True
                continue
            if ch == '{':
                depth += 1
                if depth == 1:
                    body_start = i + 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and body_start != -1:
                    return (body_start, i)
        return None

    def _inject_into_block(self, stmts: List[str]) -> List[str]:
        """Pick a random compound block from stmts, inject 1-3 random atomic
        statements (also from stmts) into its body at a random position.
        Returns the modified statement list."""
        candidates = []
        atomic = []
        for idx, s in enumerate(stmts):
            if self._PHP_BLOCK_HEAD_RE.match(s) and '{' in s:
                span = self._find_outermost_brace_body(s)
                if span:
                    body_start, body_end = span
                    body = s[body_start:body_end].strip()
                    if len(body) > 5:
                        candidates.append((idx, body_start, body_end))
            else:
                if s.strip():
                    atomic.append((idx, s))

        if not candidates or not atomic:
            return stmts

        target_idx, body_start, body_end = random.choice(candidates)
        target = stmts[target_idx]
        body = target[body_start:body_end]

        # Pick 1-3 atomic statements (not the target itself) to inject
        donors = [(i, s) for i, s in atomic if i != target_idx]
        if not donors:
            return stmts
        n_inject = min(random.randint(1, 3), len(donors))
        chosen = random.sample(donors, n_inject)
        inject_stmts = [s for _, s in chosen]

        # Find injection point: split body into lines, pick a random line boundary
        body_lines = body.split('\n')
        insert_pos = random.randint(0, len(body_lines))

        # Detect indentation from existing body lines
        indent = "    "
        for ln in body_lines:
            stripped = ln.lstrip()
            if stripped:
                indent = ln[:len(ln) - len(stripped)]
                break

        injected = [indent + s.strip() for s in inject_stmts]
        new_body_lines = body_lines[:insert_pos] + injected + body_lines[insert_pos:]
        new_body = '\n'.join(new_body_lines)

        new_stmt = target[:body_start] + new_body + target[body_end:]
        result = list(stmts)
        result[target_idx] = new_stmt
        return result

    def _statement_fuse(self, clean1: str, clean2: str,
                        vars1: List[str], vars2: List[str]) -> str:
        """Statement fusion: split both seeds into statements, interleave
        via dependency-graph topological sort with similarity tie-breaking,
        optionally inject statements into compound block bodies,
        then cross-replace one variable."""
        stmts_a = self._split_statements(clean1)
        stmts_b = self._split_statements(clean2)

        if not stmts_a and not stmts_b:
            return clean1 + '\n' + clean2

        interleaved = self._dependency_graph_interleave(stmts_a, stmts_b)

        # Block injection pass: inject atomic statements into a compound block body
        if random.random() < 0.3:
            interleaved = self._inject_into_block(interleaved)

        fused_code = '\n'.join(interleaved)

        # Cross-replace one variable from B with one from A
        fused_code = self._stmt_cross_replace_variable(fused_code, vars1, vars2)

        return fused_code

    # Matches goto jump and label statements that break when merged across seeds.
    _PHP_GOTO_RE = re.compile(r'^\s*goto\s+\w+\s*;.*$', re.M)
    _PHP_LABEL_RE = re.compile(r'^\s*\w+\s*:\s*$', re.M)

    def clean_php_header_tail(self, phpcode):
        s = phpcode.strip()
        for tag in ("===DONE===", "==DONE==", "Done"):
            if s.endswith(tag): s = s[: -len(tag)]
        s = s.strip()
        if s.startswith('<?php'): s = s[5:].lstrip()
        if s.endswith('?>'):      s = s[:-2].rstrip()
        # Remove goto/label pairs — labels from one seed become dangling when
        # the corresponding goto ends up in the other seed after fusion.
        s = self._PHP_GOTO_RE.sub('', s)
        s = self._PHP_LABEL_RE.sub('', s)
        return '\n' + s + '\n'

    def adhoc_syntax_patch(self, phpt):
        phpt = phpt.replace('echo "Done"\n', 'echo "Done";\n')
        return phpt

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        mode = 'stmt_ab' if self.stmt_fusion else 'df_ab'
        return self._build_fused_test(parent_a, parent_b, mode)

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """Produce both A->B and B->A variants for the active fusion kind."""
        if self.stmt_fusion:
            return [
                self._build_fused_test(parent_a, parent_b, 'stmt_ab'),
                self._build_fused_test(parent_a, parent_b, 'stmt_ba'),
            ]
        return [
            self._build_fused_test(parent_a, parent_b, 'df_ab'),
            self._build_fused_test(parent_a, parent_b, 'df_ba'),
        ]

    def _build_fused_test(self, parent_a, parent_b, mode):
        """Build a single fused test for a specific mode.
        mode is one of: 'stmt_ab', 'stmt_ba', 'df_ab', 'df_ba'."""
        phpcode1 = parent_a.content
        phpcode2 = parent_b.content
        meta1 = parent_a.metadata
        meta2 = parent_b.metadata
        variable1 = meta1.get('variables', [])
        variable2 = meta2.get('variables', [])
        dataflow1 = meta1.get('dataflows', [])
        dataflow2 = meta2.get('dataflows', [])
        if self.mutation:
            phpcode1 = self.mut.mutate(phpcode1)
            phpcode2 = self.mut.mutate(phpcode2)
        clean1 = self.clean_php_header_tail(phpcode1)
        clean2 = self.clean_php_header_tail(phpcode2)

        preamble1, clean1 = self._extract_preamble(clean1)
        preamble2, clean2 = self._extract_preamble(clean2)
        preamble_lines = list(dict.fromkeys(preamble1 + preamble2))
        preamble_code = '\n'.join(preamble_lines)

        clean2 = self._resolve_name_conflicts(clean1, clean2)

        _pre_cls = ""
        _after_cls = ""
        extra_class_flows = []
        all_vars = variable1 + variable2 + ['$fusion']
        if random.random() < 0.2:
            _pre_cls, _after_cls, class_vars = self._instrumentation_classfuzz(all_vars)
            if class_vars:
                extra_class_flows = [class_vars]
                all_vars.extend(class_vars)

        if mode == 'stmt_ab':
            fused_body = self._statement_fuse(clean1, clean2, variable1, variable2)
            inner = f"{_pre_cls}\n{fused_body}\n"
        elif mode == 'stmt_ba':
            fused_body = self._statement_fuse(clean2, clean1, variable2, variable1)
            inner = f"{_pre_cls}\n{fused_body}\n"
        elif mode == 'df_ab':
            new_code1, new_code2 = self.interleave_code_blocks(
                clean1, clean2, dataflow1, dataflow2,
                extra_flows=extra_class_flows)
            inner = f"{_pre_cls}\n{new_code1}\n{new_code2}\n"
        elif mode == 'df_ba':
            new_code2, new_code1 = self.interleave_code_blocks(
                clean2, clean1, dataflow2, dataflow1,
                extra_flows=extra_class_flows)
            inner = f"{_pre_cls}\n{new_code1}\n{new_code2}\n"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        _inst_api = ""
        if self.apifuzz and random.random() < 0.2:
            _inst_api = self._instrumentation_apifuzz(all_vars)
        _inst_dump = "\nvar_dump(get_defined_vars());\n"

        inner += f"{_inst_dump}\n{_inst_api}\n{_after_cls}"
        php_body = f"{preamble_code}\ntry {{\n{inner}\n}} catch (\\Throwable $_ffl_e) {{}}\n"
        fused_file = f"\n--FILE--\n<?php\n{php_body}"
        desc = f"--TEST--\nFused {parent_a.id} + {parent_b.id} ({mode})\n"
        conf = f"\n--INI--\n{meta1.get('configuration','')}\n{meta2.get('configuration','')}\n{self.random_inis()}\n"
        ext = ""
        if meta1.get('extension') or meta2.get('extension'):
            ext = f"\n--EXTENSION--\n{meta1.get('extension','')}\n{meta2.get('extension','')}\n"
        expect = "\n--EXPECT--\nthis is a flowfusion test\n"
        fused_test = f"{desc}{conf}{ext}{fused_file}{expect}"
        fused_test = re.sub("\n+", "\n", fused_test)
        fused_test = self.adhoc_syntax_patch(fused_test)
        return Seed(content=fused_test, metadata={
            "parents": [parent_a.id, parent_b.id],
            "type": "phpt",
            "mode": mode,
            "description": f"Fused {parent_a.id} + {parent_b.id} ({mode})",
        })


class PHPStateFusionStrategy(PHPFusionStrategy):
    """
    State fusion for PHP (core/state_analysis.py): grafts one seed's
    continuation into the other's state at a profiled *most complex state*
    (resource release / type conversion / exception boundary) rather than
    bridging a single value through --dataflow-fusion's shared variable,
    or interleaving whole statements by dependency graph like PHP's
    existing --state-fusion mode. Picks an intermediate program point in
    each seed instead of only ever combining seeds at their final state,
    per the design's "richer search space" argument for state fusion.
    """

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint

        host_code = self.clean_php_header_tail(host.content)
        donor_code = self.clean_php_header_tail(donor.content)
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)
        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        host_point = pick_state_point(host_code, "php", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_code, "php", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_code.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        fused_body = graft_continuation(host_code, donor_code, host_point, donor_point)
        php_body = f"try {{\n{fused_body}\n}} catch (\\Throwable $_ffl_e) {{}}\n"
        fused_file = f"\n--FILE--\n<?php\n{php_body}"
        desc = f"--TEST--\nState-fused {host.id} <- {donor.id} ({direction}, {host_point.category})\n"
        expect = "\n--EXPECT--\nthis is a flowfusion test\n"
        fused_test = re.sub("\n+", "\n", f"{desc}{fused_file}{expect}")
        fused_test = self.adhoc_syntax_patch(fused_test)

        return Seed(content=fused_test, metadata={
            "parents": [host.id, donor.id],
            "type": "phpt",
            "mode": f"state_{host_point.category}_{direction}",
            "description": f"State-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


class PHPDeclarationFusionStrategy(PHPFusionStrategy):
    """
    Declaration fusion for PHP: makes an extensible declaration expression
    in one seed refer to a declaration in the other, rather than bridging
    a runtime value. Two primitives:

    - base_ref: inject a donor class/interface name into a host class's
      `implements` list (or interface's `extends` list, which is the only
      form PHP allows for interface-to-interface inheritance).
    - trait_use: insert a `use DonorName;` as the first statement in a
      host class/trait body.

    Both can produce a "fatal error" purely from the class declaration
    executing (unresolvable interface, `use` on a non-trait, incompatible
    method signatures) — no call into the class is needed, matching the
    paper's declare/compile-time trigger for declaration fusion. The
    donor's declaration is placed first so it's defined by the time the
    host's class statement executes (PHP resolves `implements`/`extends`
    when the class declaration runs, same top-to-bottom order as any
    other statement).
    """

    _PHP_CLASS_HEADER_RE = re.compile(
        r'\b(?P<kind>class|interface|trait)\s+(?P<name>[A-Za-z_]\w*)'
        r'(?P<extends>\s+extends\s+[A-Za-z_][\w\\,\s]*?)?'
        r'(?P<implements>\s+implements\s+[A-Za-z_][\w\\,\s]*?)?'
        r'\s*\{'
    )

    @staticmethod
    def _matching_brace(text: str, mask: List[bool], open_pos: int):
        depth = 0
        for i in range(open_pos, len(text)):
            if i >= len(mask) or not mask[i]:
                continue
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return i
        return None

    def _inject_base_ref(self, code: str, donor_name: str):
        matches = list(self._PHP_CLASS_HEADER_RE.finditer(code))
        if not matches:
            return code, False
        m = random.choice(matches)
        brace_pos = m.end() - 1
        kind = m.group('kind')
        if kind == 'trait':
            return code, False  # traits can't extend/implement
        group = 'extends' if kind == 'interface' else 'implements'
        keyword = 'extends' if kind == 'interface' else 'implements'
        if m.group(group):
            new_code = code[:brace_pos].rstrip() + f", {donor_name} " + code[brace_pos:]
        else:
            new_code = code[:brace_pos].rstrip() + f" {keyword} {donor_name} " + code[brace_pos:]
        return new_code, True

    def _inject_trait_use(self, code: str, donor_name: str):
        from .state_analysis import _lexical_mask
        mask = _lexical_mask(code, 'php')
        matches = [m for m in self._PHP_CLASS_HEADER_RE.finditer(code) if m.group('kind') in ('class', 'trait')]
        if not matches:
            return code, False
        m = random.choice(matches)
        brace_pos = m.end() - 1
        close = self._matching_brace(code, mask, brace_pos)
        if close is None:
            return code, False
        insert_at = brace_pos + 1
        new_code = code[:insert_at] + f"\n    use {donor_name};\n" + code[insert_at:]
        return new_code, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_code = self.clean_php_header_tail(host.content)
        donor_code = self.clean_php_header_tail(donor.content)
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)
        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        donor_names = [m.group('name') for m in self._PHP_CLASS_HEADER_RE.finditer(donor_code)]
        fused_host = host_code
        technique = 'none'
        if donor_names:
            donor_name = random.choice(donor_names)
            for candidate in random.sample(['base_ref', 'trait_use'], k=2):
                if candidate == 'base_ref':
                    fused_host, applied = self._inject_base_ref(host_code, donor_name)
                else:
                    fused_host, applied = self._inject_trait_use(host_code, donor_name)
                if applied:
                    technique = candidate
                    break
            else:
                fused_host = host_code

        # Donor declarations first: PHP resolves extends/implements/use
        # when the class statement executes, top-to-bottom like any other
        # statement, so the referenced declaration must run first.
        inner = f"{donor_code}\n{fused_host}\n"
        php_body = f"try {{\n{inner}\n}} catch (\\Throwable $_ffl_e) {{}}\n"
        fused_file = f"\n--FILE--\n<?php\n{php_body}"
        desc = f"--TEST--\nDeclaration-fused {host.id} <- {donor.id} ({direction}, {technique})\n"
        expect = "\n--EXPECT--\nthis is a flowfusion test\n"
        fused_test = re.sub("\n+", "\n", f"{desc}{fused_file}{expect}")
        fused_test = self.adhoc_syntax_patch(fused_test)

        return Seed(content=fused_test, metadata={
            "parents": [host.id, donor.id],
            "type": "phpt",
            "mode": f"decl_{technique}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


# ==========================================
# CPython Specific Fusion Strategy
# ==========================================

class CPythonFusionStrategy(FusionStrategy):
    def __init__(self, project_root="projects/cpython", lightweight: bool = False):
        self.project_root = project_root
        self.mutation = True
        self.bridge_var_name = "fusion"
        self.mut = CPythonMutator()
        # Dataflow fusion here is already the on-the-fly, purely-random
        # scheme --pre-analysis-off calls for (no cached metadata
        # dependency), so this flag is a no-op for fuse() — it only exists
        # so CPythonStateFusionStrategy (which inherits this __init__) can
        # read self.lightweight for its own pick_state_point() calls.
        self.lightweight = lightweight

    def _extract_imports_and_body(self, code):
        imports = []
        body_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
            else:
                body_lines.append(line)
        return "\n".join(body_lines), imports

    def _instrumentation_builtins(self, defined_vars):
        if not defined_vars: return ""
        builtins = ["len", "str", "int", "bool", "type", "repr", "dir", "id", "hash", "list", "tuple", "set"]
        instruments = []
        vars_pool = defined_vars
        for _ in range(5):
            func = random.choice(builtins)
            var = random.choice(vars_pool)
            stmt = f"try: {func}({var})\nexcept: pass"
            instruments.append(stmt)
        return "\n".join(instruments) + "\n"

    def _instrumentation_bug_primitives_cpython(self, defined_vars):
        """
        (A+B)+C: Phase-directed bug primitives for CPython.
        Each primitive targets a specific CPython pipeline phase,
        using fused variables as input to make the probe meaningful.

        Phases targeted:
          P1. Peephole optimizer & constant folding
          P2a. Reference counting & UAF proxy
          P2b. GC cycle collection
          P3. Adaptive specialization & JIT boundary (CPython 3.11+)
          P4. Object model & runtime internals
        """
        if not defined_vars:
            return ""

        v1 = random.choice(defined_vars)
        v2 = random.choice(defined_vars)
        v3 = random.choice(defined_vars)

        # --- P1. Peephole Optimizer & Constant Folding ---
        # Targets: compile(), eval(), peephole pass, bytecode via dis.
        # Type-unstable fused variables break constant folding assumptions.
        p1 = f"""
# C: P1 - Peephole optimizer & constant folding
import dis, sys
try:
    _p1_v = {v1}
    # Oscillate type to stress type assumptions in folded expressions
    try: _p1_v = int(_p1_v)
    except: pass
    try: _p1_v = float(_p1_v)
    except: pass
    try: _p1_v = str(_p1_v)
    except: pass
    try: _p1_v = bool(_p1_v)
    except: pass
    try: _p1_v = list(_p1_v) if hasattr(_p1_v, '__iter__') else [_p1_v]
    except: pass

    # Probe constant folding: optimizer may incorrectly fold these
    # as compile-time constants despite type instability
    _p1_const = int({v1} or 0) + int({v2} or 0)
    _p1_fold  = _p1_const * 0        # optimizer may fold to 0 incorrectly
    _p1_dead  = (_p1_fold == 1)      # dead branch — optimizer may wrongly eliminate

    # Boundary arithmetic using fused values
    _p1_max  = sys.maxsize + int({v2} or 0)
    _p1_min  = -sys.maxsize - 1 - abs(int({v3} or 0))

    # Compile and inspect bytecode of a dynamically built expression
    # using fused values — probes the compiler's handling of live variables
    _p1_expr = "lambda x: x + " + str(int({v1} or 0))
    _p1_fn   = eval(_p1_expr)
    dis.dis(_p1_fn)
    print(_p1_v, _p1_fold, _p1_dead, _p1_max, _p1_min)
except Exception as _e:
    print(_e)
"""

        # --- P2a. Reference Counting & UAF Proxy ---
        # Targets: Py_INCREF/Py_DECREF logic, tp_dealloc, weakref callbacks.
        # Drops aliasing chain in a specific order to probe whether refcount
        # reaches zero prematurely while another path still holds a pointer.
        p2a = f"""
# C: P2a - Reference counting / UAF proxy
import sys, weakref
try:
    _p2_orig = {v1}
    print("refcount before aliasing:", sys.getrefcount(_p2_orig))

    # Build aliasing chain on fused variable
    _p2_ref1 = _p2_orig          # refcount +1
    _p2_ref2 = _p2_ref1          # refcount +1
    _p2_copy = _p2_orig          # soft copy
    print("refcount after aliasing:", sys.getrefcount(_p2_orig))

    # Attach a weakref to probe deallocation timing
    try:
        _p2_weak = weakref.ref(_p2_orig, lambda r: print("weakref callback fired"))
    except TypeError:
        _p2_weak = None           # not all types support weakref

    # Drop in reverse order — probes premature deallocation
    del _p2_orig
    del _p2_ref1
    _p2_read = _p2_ref2           # read through last alias
    _p2_ref2 = {v2}               # write through — probes stale pointer
    print("weakref alive:", _p2_weak() if _p2_weak else "N/A")
    print("p2_read:", _p2_read)
except Exception as _e:
    print(_e)
"""

        # --- P2b. GC Cycle Collection ---
        # Targets: gc.collect(), tp_traverse, tp_clear, finalizer ordering.
        # Builds heterogeneous cross-seed cycles to stress the collector
        # on object graphs it has never seen before.
        p2b = f"""
# C: P2b - GC cycle collection
import gc
try:
    class _FuzzNode:
        def __init__(self, val):
            self.val  = val
            self.next = None
            self.prev = None
        def __del__(self):
            pass  # finalizer during GC — probes finalizer ordering

    _p2_nodeA      = _FuzzNode({v1})
    _p2_nodeB      = _FuzzNode({v2})
    _p2_nodeA.next = _p2_nodeB        # cross-seed edge
    _p2_nodeB.prev = _p2_nodeA        # back edge — cycle
    _p2_nodeA.self = _p2_nodeA        # self-reference
    _p2_nodeB.data = {v3}             # attach fused value to cycle node

    # Trigger cycle collector on a non-trivial heterogeneous graph
    del _p2_nodeA, _p2_nodeB
    _collected = gc.collect()
    print("GC collected:", _collected)
except Exception as _e:
    print(_e)
"""

        # --- P3. Adaptive Specialization & JIT Boundary (CPython 3.11+) ---
        # Targets: LOAD_ATTR, CALL, BINARY_OP specialization/de-specialization,
        # and the tier-1 to tier-2 JIT boundary introduced in CPython 3.13.
        # Warms up the specializer with a consistent type, then violates
        # its recorded assumptions to force de-specialization.
        p3 = f"""
# C: P3 - Adaptive specialization & JIT boundary
try:
    class _FuzzSpecialize:
        def __init__(self, val):
            self.val = val
        def method(self, x):
            return self.val

    # Warm-up phase: feed consistent type to train the specializer
    # CALL and LOAD_ATTR will specialize for _FuzzSpecialize
    _p3_warm = _FuzzSpecialize({v1})
    for _i in range(100):
        _ = _p3_warm.method({v2})
        _ = _p3_warm.val

    # Violation phase: swap in incompatible types to trigger de-specialization
    for _p3_v in [int({v1} or 0), str({v2} or ''), None, {v3}, [], {{}}]:
        try:
            _ = _p3_warm.method(_p3_v)
        except Exception:
            pass

    # Tier boundary probe: oscillate between hot and cold paths
    for _i in range(200):
        try:
            _p3_obj = _FuzzSpecialize({v1}) if _i % 3 == 0 \
                 else (int({v2} or 0)       if _i % 3 == 1 \
                 else  str({v3} or ''))
            _p3_obj.method({v1}) if hasattr(_p3_obj, 'method') else str(_p3_obj)
        except Exception:
            pass  # cold path — probes exception handling in specialized frames

    # BINARY_OP specialization with type-unstable fused values
    _p3_a = {v1}
    _p3_b = {v2}
    for _p3_type in [int, float, str, bool]:
        try:
            _p3_result = _p3_type(_p3_a or 0) + _p3_type(_p3_b or 0)
            print(_p3_result)
        except Exception:
            pass
except Exception as _e:
    print(_e)
"""

        # --- P4. Object Model & Runtime Internals ---
        # Targets: copy protocol, pickle round-trips, __slots__ descriptor
        # stress, and __init_subclass__ type machinery — all of which
        # exercise deep object model logic rarely reached by normal execution.
        p4 = f"""
# C: P4 - Object model & runtime internals
import copy, pickle
try:
    _p4_obj = {v1}

    # Deep copy chain: mutations on copies should not affect the original
    try:
        _p4_c1 = copy.copy(_p4_obj)
        _p4_c2 = copy.deepcopy(_p4_obj)
        print("copy eq:", _p4_obj == _p4_c1)
        print("deepcopy eq:", _p4_obj == _p4_c2)
        del _p4_c1
    except Exception as _ce:
        print("copy error:", _ce)

    # Pickle round-trip: tests that fused object graph is internally
    # consistent across all supported protocol versions
    try:
        for _proto in range(pickle.HIGHEST_PROTOCOL + 1):
            _p4_ser   = pickle.dumps(_p4_obj, protocol=_proto)
            _p4_deser = pickle.loads(_p4_ser)
            print(f"pickle proto {{_proto}} eq:", _p4_deser == _p4_obj)
    except Exception as _pe:
        print("pickle error:", _pe)

    # __slots__ descriptor stress: probe descriptor protocol
    # (__get__, __set__, __delete__) using fused values
    try:
        class _FuzzSlot:
            __slots__ = ['a', 'b']
        _p4_slot   = _FuzzSlot()
        _p4_slot.a = {v2}
        _p4_slot.b = {v3}
        print("slot a:", _p4_slot.a, "slot b:", _p4_slot.b)
        del _p4_slot.a
    except Exception as _se:
        print("slot error:", _se)

    # __init_subclass__ stress: inject fused value into subclass
    # at creation time to probe the type machinery
    try:
        class _FuzzBase:
            def __init_subclass__(cls, **kwargs):
                super().__init_subclass__(**kwargs)
                cls.fused = {v1}
        class _FuzzChild(_FuzzBase):
            pass
        print("subclass fused:", _FuzzChild.fused)
    except Exception as _ie:
        print("subclass error:", _ie)

except Exception as _e:
    print(_e)
"""

        # --- P5. Frame & Code Object Manipulation ---
        # Targets: _PyEval_EvalFrameDefault, frame locals dict, types.CodeType.
        # CPython exposes live frame objects via sys._getframe() — mutating
        # f_locals or replacing a function's __code__ at runtime exercises
        # paths the interpreter assumes are read-only during execution.
        p5 = f"""
# C: P5 - Frame & code object manipulation
import sys, types, dis
try:
    # Capture the live frame and mutate f_locals with fused values
    _p5_frame = sys._getframe(0)
    _p5_frame.f_locals['_p5_injected'] = {v1}   # inject fused var into live frame
    print("injected into frame:", _p5_frame.f_locals.get('_p5_injected'))

    # Build a new code object by replacing the constants tuple
    # of a compiled function with fused-value-derived constants.
    # Probes whether the interpreter handles unexpected constant types.
    def _p5_target(x):
        return x + 1

    _p5_co       = _p5_target.__code__
    _p5_new_consts = tuple(
        {v1} if isinstance(c, int) else c
        for c in _p5_co.co_consts
    )
    try:
        _p5_new_co = _p5_co.replace(co_consts=_p5_new_consts)
        _p5_target.__code__ = _p5_new_co        # hot-swap code object
        print("hot-swapped result:", _p5_target({v2}))
    except Exception as _ce:
        print("code replace error:", _ce)

    # Inspect bytecode of a fused-value lambda — probes line number
    # table and co_linetable encoding for dynamically built code
    _p5_fn = eval("lambda x: x + " + str(int({v2} or 0)))
    dis.dis(_p5_fn)
    print("co_consts:", _p5_fn.__code__.co_consts)
    print("co_varnames:", _p5_fn.__code__.co_varnames)
except Exception as _e:
    print(_e)
"""

        # --- P6. Generator & Coroutine Frame Suspension ---
        # Targets: frame suspension/resumption logic, gi_frame lifecycle,
        # GeneratorExit propagation, and async coroutine __await__ protocol.
        # Generators suspend a live frame mid-execution — sending fused values
        # into a suspended frame probes the interpreter's ability to correctly
        # restore register state and local variables across yield points.
        p6 = f"""
# C: P6 - Generator & coroutine frame suspension
try:
    def _p6_gen(seed_val):
        val = seed_val
        while True:
            received = yield val       # suspend — frame frozen here
            val = received if received is not None else val

    _p6_g = _p6_gen({v1})
    _p6_first = next(_p6_g)
    print("gen initial:", _p6_first)

    # Send fused values into a suspended frame — probes frame restoration
    for _p6_v in [{v1}, {v2}, {v3}, None, 0, ""]:
        try:
            _p6_out = _p6_g.send(_p6_v)
            print("gen send:", _p6_out)
        except StopIteration:
            break
        except Exception as _ge:
            print("gen send error:", _ge)

    # Inject an exception into the suspended frame via throw()
    # probes GeneratorExit handling and frame cleanup
    try:
        _p6_g.throw(ValueError, ValueError("fuzz throw"))
    except (ValueError, StopIteration) as _te:
        print("gen throw caught:", _te)

    # Coroutine __await__ protocol stress
    import asyncio
    async def _p6_coro(val):
        await asyncio.sleep(0)
        return val

    async def _p6_runner():
        # Run coroutine with fused values as inputs
        for _v in [{v1}, {v2}, {v3}]:
            try:
                _result = await _p6_coro(_v)
                print("coro result:", _result)
            except Exception as _ae:
                print("coro error:", _ae)

    try:
        asyncio.run(_p6_runner())
    except Exception as _re:
        print("asyncio error:", _re)
except Exception as _e:
    print(_e)
"""

        # --- P7. Metaclass & MRO Stress ---
        # Targets: C3 MRO linearization, type.__new__, tp_mro, __set_name__,
        # __init_subclass__, and the descriptor protocol across class hierarchies.
        # Dynamic class creation with complex multiple inheritance exposes
        # edge cases in the MRO algorithm and class creation machinery
        # that static code rarely triggers.
        p7 = f"""
# C: P7 - Metaclass & MRO stress
try:
    # Inject fused value into class namespace via __prepare__
    class _FuzzMeta(type):
        def __prepare__(mcs, name, bases, **kwargs):
            ns = super().__prepare__(name, bases, **kwargs)
            ns['fused'] = {v1}           # inject fused value at class creation
            return ns
        def __new__(mcs, name, bases, ns, **kwargs):
            ns['fused_new'] = {v2}       # inject again at __new__
            return super().__new__(mcs, name, bases, ns)
        def __init__(cls, name, bases, ns, **kwargs):
            super().__init__(name, bases, ns)
            cls.fused_init = {v3}        # inject at __init__

    # Diamond inheritance — stresses C3 MRO with metaclass
    class _A(metaclass=_FuzzMeta): pass
    class _B(_A): pass
    class _C(_A): pass
    try:
        class _D(_B, _C): pass           # diamond: MRO must linearize correctly
        print("MRO:", [c.__name__ for c in _D.__mro__])
        print("fused:", _D.fused, _D.fused_new, _D.fused_init)
    except TypeError as _mro_e:
        print("MRO error:", _mro_e)      # inconsistent hierarchy — expected

    # __set_name__ stress: descriptor injected into dynamically created class
    class _FuzzDescriptor:
        def __set_name__(self, owner, name):
            self.name = name
            self.owner_fused = {v1}
        def __get__(self, obj, objtype=None):
            return self.owner_fused if obj is None else {v2}
        def __set__(self, obj, value):
            pass

    _DynCls = type('_DynCls', (), {{'attr': _FuzzDescriptor()}})
    print("descriptor get:", _DynCls.attr, _DynCls().attr)
except Exception as _e:
    print(_e)
"""

        # --- P8. Tracing & Profiling Hook Interference ---
        # Targets: interaction between sys.settrace and the adaptive specializing
        # interpreter. CPython disables specialization when a trace hook is active
        # — installing and removing hooks mid-execution forces repeated
        # transitions between traced (unspecialized) and untraced (specialized)
        # modes, probing the consistency of bytecode state across these switches.
        p8 = f"""
# C: P8 - Tracing hook interference with specialization
import sys
try:
    _p8_trace_events = []

    def _p8_tracer(frame, event, arg):
        _p8_trace_events.append((event, frame.f_lineno))
        return _p8_tracer     # return self to keep tracing

    # Warm up specialization without a trace hook
    class _FuzzTrace:
        def __init__(self, v): self.v = v
        def method(self, x):   return self.v

    _p8_obj = _FuzzTrace({v1})
    for _i in range(50):
        _ = _p8_obj.method({v2})   # specializes CALL + LOAD_ATTR

    # Install trace hook mid-execution — forces de-specialization
    sys.settrace(_p8_tracer)
    for _i in range(20):
        _ = _p8_obj.method({v2})   # now runs unspecialized under trace

    # Remove hook — interpreter must re-specialize cleanly
    sys.settrace(None)
    for _i in range(50):
        _ = _p8_obj.method({v2})   # should re-specialize

    # Oscillate hook on/off to maximally stress the mode transition
    for _i in range(10):
        sys.settrace(_p8_tracer if _i % 2 == 0 else None)
        _ = _p8_obj.method({v3})

    sys.settrace(None)             # always clean up
    print("trace events captured:", len(_p8_trace_events))
except Exception as _e:
    sys.settrace(None)
    print(_e)
"""

        # --- P9. Buffer Protocol & Memoryview ---
        # Targets: tp_as_buffer, PyBUF_* flags, memoryview slicing and casting,
        # and struct pack/unpack with fused-value-derived format strings.
        # The buffer protocol is a low-level interface that bypasses normal
        # Python object semantics — errors here typically surface as segfaults
        # or assertion failures rather than Python exceptions.
        p9 = f"""
# C: P9 - Buffer protocol & memoryview
import array, struct, ctypes
try:
    # Build a typed array from a fused value and probe buffer views
    _p9_size = max(1, abs(int({v1} or 1)) % 256)
    _p9_arr  = array.array('i', range(_p9_size))
    _p9_mv   = memoryview(_p9_arr)

    # Slice and cast the memoryview — probes buffer shape/strides
    _p9_slice = _p9_mv[::2] if len(_p9_mv) > 1 else _p9_mv
    try:
        _p9_cast = _p9_mv.cast('B')    # reinterpret as bytes
        print("cast itemsize:", _p9_cast.itemsize, "len:", len(_p9_cast))
    except Exception as _ce:
        print("cast error:", _ce)

    # struct pack/unpack with fused-value-derived data
    # probes struct module's handling of boundary values
    for _fmt, _val in [
        ('i',  int({v1} or 0) % (2**31)),
        ('f',  float({v2} or 0.0)),
        ('?',  bool({v3})),
        ('q',  int({v1} or 0)),
    ]:
        try:
            _p9_packed   = struct.pack(_fmt, _val)
            _p9_unpacked = struct.unpack(_fmt, _p9_packed)
            print(f"struct {{_fmt}}:", _p9_unpacked)
        except Exception as _se:
            print(f"struct error {{_fmt}}:", _se)

    # Write fused value into a ctypes buffer — probes raw memory write path
    try:
        _p9_buf = (ctypes.c_int * _p9_size)()
        _p9_buf[0] = int({v2} or 0) % (2**31)
        print("ctypes buf[0]:", _p9_buf[0])
    except Exception as _be:
        print("ctypes error:", _be)
except Exception as _e:
    print(_e)
"""

        # --- P10. Closure & Nonlocal Cell Mutation ---
        # Targets: LOAD_DEREF / STORE_DEREF opcodes, cell object lifecycle,
        # and the interaction between closures and the specializing interpreter.
        # Closures capture variables as cell objects — mutating a cell from
        # outside the closure (via __closure__) while the inner function is
        # executing probes whether the interpreter correctly reads through
        # the cell indirection under specialization.
        p10 = f"""
# C: P10 - Closure & nonlocal cell mutation
import ctypes
try:
    _p10_cell_val = {v1}

    def _p10_outer(init):
        _p10_x = init              # captured as cell object
        def _p10_inner():
            nonlocal _p10_x
            _p10_x = {v2}          # STORE_DEREF — writes through cell
            return _p10_x          # LOAD_DEREF — reads through cell
        return _p10_inner, lambda: _p10_x  # expose cell reader too

    _p10_inner, _p10_reader = _p10_outer(_p10_cell_val)

    # Warm up the inner function so the specializer sees LOAD_DEREF
    for _i in range(50):
        _ = _p10_inner()

    # Directly mutate the cell object from outside the closure
    # to create a state the specializer has never seen
    try:
        _p10_cell = _p10_inner.__closure__[0]
        _p10_cell.cell_contents = {v3}     # overwrite cell from outside
        print("cell after mutation:", _p10_reader())
        print("inner after mutation:", _p10_inner())
    except (ValueError, AttributeError) as _mut_e:
        print("cell mutation error:", _mut_e)

    # Nested closure depth stress — each level adds a cell indirection
    def _make_deep(depth, val):
        if depth == 0:
            return lambda: val
        inner = _make_deep(depth - 1, val)
        def _wrap():
            return inner()         # chains LOAD_DEREF across N frames
        return _wrap

    _p10_deep = _make_deep(min(abs(int({v1} or 1)) % 20 + 2, 20), {v2})
    print("deep closure:", _p10_deep())
except Exception as _e:
    print(_e)
"""


        phases = [p1, p2a, p2b, p3, p4, p5, p6, p7, p8, p9, p10]
        selected = random.choice(phases)
        return '\n' + selected + '\n'

    def _splice_functions_or_classes(self, code1, code2, fusion_rhs="0"):
        def get_blocks(code):
            blocks = []
            imports = []
            current_block = []
            for line in code.splitlines():
                if line.startswith("import ") or line.startswith("from "):
                    imports.append(line)
                elif line.strip() and not line.startswith(" "):
                    if current_block:
                        block_str = "\n".join(current_block)
                        if block_str.strip().endswith(":"): block_str += "\n    pass"
                        blocks.append(block_str)
                        current_block = []
                    current_block.append(line)
                elif line.strip():
                    current_block.append(line)
            if current_block:
                block_str = "\n".join(current_block)
                if block_str.strip().endswith(":"): block_str += "\n    pass"
                blocks.append(block_str)
            return imports, blocks

        imports1, blocks1 = get_blocks(code1)
        imports2, blocks2 = get_blocks(code2)
        bridge_def = f"\n# --- Fusion Bridge ---\nfusion = {fusion_rhs}\n"
        final_content = "\n".join(blocks1) + bridge_def + "\n".join(blocks2)
        return final_content

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        sa = parent_a.content
        sb = parent_b.content

        if self.mutation:
            sa = self.mut.mutate(sa)
            sb = self.mut.mutate(sb)

        a_body, a_imports = self._extract_imports_and_body(sa)
        b_body, b_imports = self._extract_imports_and_body(sb)
        all_imports = sorted(list(set(a_imports + b_imports)))

        # CPython is dynamically typed, so a "same type" or "defined vs.
        # undefined" check on the bridge pair buys little over chance —
        # pick the A-side producer and the B-side substitution site
        # uniformly at random from each side's candidate names.
        a_candidates = collect_top_level_assigned_vars(a_body)
        b_candidates = collect_bare_vars(b_body)

        a_var    = random.choice(a_candidates) if a_candidates else None
        b_choice = random.choice(b_candidates) if b_candidates else None

        fusion_rhs = a_var if a_var else "0"
        a_text = a_body
        b_text = b_body

        if b_choice:
            b_text, _ = replace_one_b_occurrence(b_text, b_choice, "fusion")

        final_body = self._splice_functions_or_classes(a_text, b_text, fusion_rhs)

        fut_block      = "".join(f + "\n" for f in all_imports if "future" in f)
        normal_imports = "".join(f + "\n" for f in all_imports if "future" not in f)
        final_content  = fut_block + normal_imports + "\n" + final_body

        # --- (A+B)+C: append phase-directed bug primitives ---
        # defined_vars pools both fused bridge variable and all
        # candidates from A and B so C has the richest possible state to probe
        defined_vars = list(set(
            ["fusion"]
            + (a_candidates or [])
            + (b_candidates or [])
        ))
        bug_primitives = self._instrumentation_bug_primitives_cpython(defined_vars)
        builtin_inst   = self._instrumentation_builtins(["fusion"])

        final_content += "\n" + builtin_inst
        final_content += "\n" + bug_primitives

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "python",
                "description": f"Fused {parent_a.id} + {parent_b.id}"
            }
        )


class CPythonStateFusionStrategy(CPythonFusionStrategy):
    """
    State fusion for CPython (core/state_analysis.py): grafts one seed's
    continuation into the other's state at a profiled most complex state
    (a `.close()`/`del`/context-manager exit, a type-coercion call, an
    `except`/`finally` boundary) instead of CPythonFusionStrategy's
    single-bridge-variable substitution. Complements it rather than
    replacing it — this class is opted into separately.
    """

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint

        host_src = host.content
        donor_src = donor.content
        if self.mutation:
            host_src = self.mut.mutate(host_src)
            donor_src = self.mut.mutate(donor_src)

        host_body, host_imports = self._extract_imports_and_body(host_src)
        donor_body, donor_imports = self._extract_imports_and_body(donor_src)
        all_imports = sorted(set(host_imports) | set(donor_imports))

        host_point = pick_state_point(host_body, "cpython", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_body, "cpython", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_body.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        fused_body = graft_continuation(host_body, donor_body, host_point, donor_point)
        final_content = "\n".join(all_imports) + "\n" + fused_body

        return Seed(
            content=final_content,
            metadata={
                "parents": [host.id, donor.id],
                "type": "python",
                "mode": f"state_{host_point.category}_{direction}",
                "description": f"State-fused {host.id} <- {donor.id} ({direction})",
            }
        )


class CPythonDeclarationFusionStrategy(CPythonFusionStrategy):
    """
    Declaration fusion for CPython: injects a donor-declared class as an
    additional base class in a host class's base-class list. CPython has
    no separate compile-time type check, but a `class` *statement*'s base
    tuple is still resolved the moment that statement executes — MRO
    (C3 linearization) computation, metaclass selection/conflict
    detection, and `__init_subclass__`/`__set_name__` hooks all run then,
    purely from the class being declared, before any method is ever
    called. That's the CPython analogue of the paper's declare/compile-
    time-only trigger.
    """

    _PY_CLASS_HEADER_RE = re.compile(
        r'^(?P<indent>[ \t]*)class\s+(?P<name>[A-Za-z_]\w*)\s*'
        r'(?:\((?P<bases>[^)]*)\))?'
        r'\s*:',
        re.MULTILINE,
    )

    @staticmethod
    def _insert_positional_base(bases_str: str, donor_name: str) -> str:
        """Insert `donor_name` as a new positional base. A base list can
        contain keyword args (`metaclass=Meta`) or `**kwargs`, and Python
        requires every positional argument to precede those — appending
        blindly at the end is a syntax error whenever the class already
        has one of those forms, so this inserts before the first keyword-
        like part instead of after the last part."""
        depth = 0
        parts, cur = [], ''
        for ch in bases_str:
            if ch in '([{':
                depth += 1
            elif ch in ')]}':
                depth = max(0, depth - 1)
            if ch == ',' and depth == 0:
                parts.append(cur)
                cur = ''
            else:
                cur += ch
        parts.append(cur)

        kw_idx = None
        for i, p in enumerate(parts):
            stripped = p.strip()
            if stripped.startswith('**') or re.match(r'^[A-Za-z_]\w*\s*=(?!=)', stripped):
                kw_idx = i
                break
        if kw_idx is None:
            parts.append(f' {donor_name}')
        else:
            parts.insert(kw_idx, f' {donor_name}')
        return ','.join(parts)

    def _inject_base_class(self, code: str, donor_name: str):
        # Restrict to top-level (indent=='') class headers — a nested
        # donor/host class would need dotted qualification to reference,
        # which this lightweight regex-based approach doesn't track.
        matches = [m for m in self._PY_CLASS_HEADER_RE.finditer(code) if m.group('indent') == '']
        if not matches:
            return code, False
        m = random.choice(matches)
        if m.group('bases') is not None:
            new_bases = self._insert_positional_base(m.group('bases'), donor_name)
            new_code = code[:m.start('bases')] + new_bases + code[m.end('bases'):]
        else:
            colon_pos = m.end() - 1
            new_code = code[:colon_pos] + f"({donor_name})" + code[colon_pos:]
        return new_code, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_src, donor_src = host.content, donor.content
        if self.mutation:
            host_src = self.mut.mutate(host_src)
            donor_src = self.mut.mutate(donor_src)

        host_body, host_imports = self._extract_imports_and_body(host_src)
        donor_body, donor_imports = self._extract_imports_and_body(donor_src)
        all_imports = sorted(set(host_imports) | set(donor_imports))

        donor_names = [m.group('name') for m in self._PY_CLASS_HEADER_RE.finditer(donor_body)
                        if m.group('indent') == '']
        fused_host, applied = host_body, False
        if donor_names:
            donor_name = random.choice(donor_names)
            fused_host, applied = self._inject_base_class(host_body, donor_name)

        # Donor first: a class statement's base tuple is evaluated the
        # instant that statement executes, so the referenced donor class
        # must already be defined, same top-to-bottom rule as any name.
        final_content = "\n".join(all_imports) + "\n" + donor_body + "\n" + fused_host

        return Seed(content=final_content, metadata={
            "parents": [host.id, donor.id],
            "type": "python",
            "mode": f"decl_{'base_class' if applied else 'none'}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


# ==========================================
# Swift Specific Fusion Strategy
# ==========================================

class SwiftFusionStrategy(FusionStrategy):
    """
    Swift-Specific Fusion Strategy.
    Ensures correct structure by strictly ordering imports and bodies,
    and attempting simple type-safe bridging of values.
    """
    def __init__(self, project_root="projects/swift", type_match: bool = False,
                 lightweight: bool = False):
        self.project_root = project_root
        self.bridge_var_name = "fusion"
        # --dataflow-type-match: bias the B-side bridge target toward a
        # variable with the same inferred type as the chosen A-side variable
        # 90% of the time (falling back to a random B variable when none
        # match), and pick purely at random the other 10%. Off by default —
        # behavior is unchanged from before this flag existed.
        self.type_match = type_match
        # Dataflow fusion here is already on-the-fly (_extract_vars scans
        # the seed directly, no cached metadata), so this is a no-op for
        # fuse() — it only exists so SwiftStateFusionStrategy (which
        # inherits this __init__) can read self.lightweight.
        self.lightweight = lightweight

    def _split_imports_and_body(self, code):
        """Extracts imports to ensure they can be hoisted."""
        imports = []
        body_lines = []
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import "):
                imports.append(stripped)
            else:
                body_lines.append(line)
        return "\n".join(body_lines), imports

    def _extract_vars(self, code):
        """
        Naive extraction of top-level variable definitions like 'var x = ...' or 'let x = ...'
        Returns a list of (var_name, inferred_type_hint) tuples.
        """
        # Matches: var x = 10, let y: Int = 20, var s = "string"
        # Group 2 is name, Group 3 is optional type hint, Group 4 is value
        regex = r'^\s*(var|let)\s+([a-zA-Z0-9_]+)(\s*:\s*[a-zA-Z0-9_]+)?\s*=\s*(.+)'
        vars_found = []
        for line in code.splitlines():
            match = re.match(regex, line)
            if match:
                name = match.group(2)
                val_str = match.group(4).strip()
                
                # Simple type inference based on value structure
                inferred_type = "Any"
                if val_str.isdigit():
                    inferred_type = "Int"
                elif val_str.startswith('"'):
                    inferred_type = "String"
                elif val_str == "true" or val_str == "false":
                    inferred_type = "Bool"
                elif val_str.startswith("["):
                    inferred_type = "Array"
                
                vars_found.append((name, inferred_type))
        return vars_found

    def _replace_compatible_literal(self, code, source_var_name, source_type):
        """
        Finds literal(s) in 'code' that match 'source_type' and replaces a
        weighted-random number of them (see pick_occurrence_count) with
        'source_var_name', so the bridge value can cross-contaminate more
        than one use site in B.
        """
        # Regexes for literals
        int_lit = r'(?<![a-zA-Z0-9_])\d+(?![a-zA-Z0-9_])'
        str_lit = r'"[^"]*"'
        bool_lit = r'\b(?:true|false)\b'

        target_regex = None
        if source_type == "Int": target_regex = int_lit
        elif source_type == "String": target_regex = str_lit
        elif source_type == "Bool": target_regex = bool_lit

        if not target_regex:
            return code, False

        lines = code.splitlines()
        candidates = []  # (line_idx, match)
        for i, line in enumerate(lines):
            if "//" in line or "import " in line:
                continue
            for m in re.finditer(target_regex, line):
                candidates.append((i, m))

        if not candidates:
            return code, False

        n = min(pick_occurrence_count(), len(candidates))
        chosen_by_line: Dict[int, list] = {}
        for i, m in random.sample(candidates, n):
            chosen_by_line.setdefault(i, []).append(m)

        for i, matches in chosen_by_line.items():
            line = lines[i]
            for m in sorted(matches, key=lambda mm: mm.start(), reverse=True):
                line = line[:m.start()] + source_var_name + line[m.end():]
            lines[i] = line

        return "\n".join(lines), True

    def _pick_bridge_target_var(self, v_type, b_vars):
        """Choose the name of a B-side variable to substitute with the
        bridge var. Under --dataflow-type-match: 90% of the time, prefer a
        variable whose inferred type matches `v_type`, falling back to a
        uniformly random B variable when none match; the other 10% (and
        always, when the flag isn't set) picks uniformly at random. Returns
        None if B has no extracted variables at all."""
        if not b_vars:
            return None
        if self.type_match and random.random() < 0.90:
            same_type = [n for n, t in b_vars if t == v_type]
            if same_type:
                return random.choice(same_type)
        return random.choice([n for n, _ in b_vars])

    def _bug_primitives(self, bridge_var: str, bridge_type: str) -> list:
        """
        Phase-directed bug primitives for the Swift compiler pipeline.
        Each primitive targets a distinct stress area: type inference,
        SIL verification, generics monomorphization, existentials, closures,
        ownership, concurrency, dynamic casting, and property wrappers.

        All symbols are prefixed _ffl_ to avoid collisions with seed code.
        The bridge_var is woven into each primitive so the constraint solver
        must reason about a cross-seed value of unknown-to-it provenance,
        amplifying the chance of triggering edge-case assertion failures.

        'bridge_type' drives which Swift type annotation is used for the
        bridge slot — Int, String, Bool, or Any (erased to protocol).
        """
        # Map inferred bridge type to Swift annotation used inside primitives
        _type_ann = {"Int": "Int", "String": "String", "Bool": "Bool"}.get(bridge_type, "Any")
        # A safe cast expression that works for all bridge types
        _as_int  = f"(Int(exactly: {bridge_var} as AnyObject as! NSObject as? Int ?? 0) ?? 0)" \
                   if bridge_type == "Any" else \
                   f"(Int(exactly: {bridge_var}) ?? 0)" if bridge_type == "Int" else "0"
        _bv = bridge_var  # shorthand

        # --- P1: Generic type-inference & constraint-solver stress ---
        # Targets: ConstraintSystem, associated-type deduction, where-clause
        # checking, and the generic specialisation pipeline.
        # Feeding a cross-seed bridge value into a deeply nested generic forces
        # the constraint solver to reason about a type it has no context for.
        p1 = f"""
// P1: Generic constraint solver & associated-type stress
protocol _FflEquatable {{
    associatedtype Value
    func value() -> Value
}}
struct _FflBox<T>: _FflEquatable {{
    private let _v: T
    init(_ v: T) {{ self._v = v }}
    func value() -> T {{ return _v }}
}}
struct _FflNested<Outer: _FflEquatable, Inner: _FflEquatable>
    where Outer.Value == Inner.Value {{
    let outer: Outer
    let inner: Inner
    func merged() -> Outer.Value {{ outer.value() }}
}}
func _ffl_p1_infer<T>(_ a: T, _ b: T) -> [T] {{ [a, b] }}
do {{
    let _ffl_box1 = _FflBox({_bv})
    let _ffl_box2 = _FflBox({_bv})
    let _ffl_nest = _FflNested(outer: _ffl_box1, inner: _ffl_box2)
    let _ffl_arr  = _ffl_p1_infer(_ffl_nest.merged(), _ffl_nest.outer.value())
    _ = _ffl_arr
}}
"""

        # --- P2: Existential boxing & protocol metatype stress ---
        # Targets: existential containers (type erasure), protocol metatypes,
        # and the `any`/`some` split introduced in Swift 5.7.
        # Repeated any↔concrete round-trips exercise open-existential SIL
        # lowering and protocol witness table lookup.
        p2 = f"""
// P2: Existential boxing / any-Protocol metatype stress
protocol _FflShape {{
    func area() -> Double
    var tag: String {{ get }}
}}
struct _FflCircle: _FflShape {{
    let r: Double
    func area() -> Double {{ .pi * r * r }}
    var tag: String {{ "circle" }}
}}
struct _FflRect: _FflShape {{
    let w, h: Double
    func area() -> Double {{ w * h }}
    var tag: String {{ "rect" }}
}}
func _ffl_p2_sum(_ shapes: [any _FflShape]) -> Double {{
    shapes.reduce(0.0) {{ $0 + $1.area() }}
}}
do {{
    let _ffl_r = Double(String(describing: {_bv}).count)
    let _ffl_shapes: [any _FflShape] = [_FflCircle(r: _ffl_r), _FflRect(w: _ffl_r, h: 2.0)]
    _ = _ffl_p2_sum(_ffl_shapes)
    let _ffl_meta: any _FflShape.Type = _FflCircle.self
    _ = _ffl_meta.init(r: _ffl_r)
}}
"""

        # --- P3: Opaque result types (`some`) & reverse type inference ---
        # Targets: reverse-inference of opaque return types, primary associated
        # types, and the SIL opaque-type lowering pass.
        # Returning a bridge-value-dependent concrete type through `some Protocol`
        # forces the compiler to prove type identity at the call site.
        p3 = f"""
// P3: Opaque result type (some) reverse-inference stress
protocol _FflProducer {{
    associatedtype Output
    func produce() -> Output
}}
struct _FflIntProducer: _FflProducer {{
    let seed: Int
    func produce() -> Int {{ seed &* 6364136223846793005 &+ 1442695040888963407 }}
}}
struct _FflStrProducer: _FflProducer {{
    let seed: String
    func produce() -> String {{ seed + seed }}
}}
@inlinable
func _ffl_p3_make(_ flag: Bool) -> some _FflProducer {{
    if flag {{ return _FflIntProducer(seed: 42) as! any _FflProducer as! _FflIntProducer }}
    return _FflIntProducer(seed: 0)
}}
do {{
    let _ffl_flag = String(describing: {_bv}).isEmpty
    let _ffl_prod = _ffl_p3_make(_ffl_flag)
    _ = _ffl_prod.produce()
}}
"""

        # --- P4: Closure capture, @escaping, and ownership stress ---
        # Targets: capture-list lowering, @escaping vs noescape ABI, and
        # the SIL ownership verifier's tracking of captured value lifetimes.
        # Capturing a bridge variable in nested closures of mixed escaping-ness
        # forces SIL to generate both stack and heap closures in the same function.
        p4 = f"""
// P4: Closure capture / @escaping / ownership stress
func _ffl_p4_apply<T>(_ f: () -> T) -> T {{ f() }}
func _ffl_p4_escape<T>(_ f: @escaping () -> T) -> () -> T {{ f }}
do {{
    var _ffl_cap = {_bv}
    // noescape: bridge captured by reference on the stack
    let _ffl_local = _ffl_p4_apply {{ _ffl_cap }}
    // @escaping: bridge promoted to heap box
    let _ffl_esc   = _ffl_p4_escape {{ _ffl_cap }}
    // nested closure capturing both outer and inner captures
    let _ffl_nest: () -> String = {{
        let inner = _ffl_esc()
        return "\\(inner) \\(_ffl_local)"
    }}
    _ = _ffl_nest()
    // mutation after escape — probes copy-on-write / exclusive-access
    _ffl_cap = {_bv}
    _ = _ffl_esc()
}}
"""

        # --- P5: Dynamic casting chain (as?, as!, type(of:)) ---
        # Targets: dynamic_cast SIL instruction, bridging conversions between
        # Swift and ObjC/Foundation types, and the metadata lookup machinery.
        # Chaining casts through protocol existentials and concrete types
        # exercises paths in the runtime that the type checker cannot fully
        # evaluate statically.
        p5 = f"""
// P5: Dynamic casting chain stress
protocol _FflCastable: AnyObject {{}}
class _FflBase: _FflCastable {{
    var v: Int = 0
}}
class _FflDerived: _FflBase {{
    var extra: String = ""
}}
func _ffl_p5_cast(_ obj: AnyObject) -> String {{
    if let d = obj as? _FflDerived {{ return "derived:\\(d.extra)" }}
    if let b = obj as? _FflBase    {{ return "base:\\(b.v)" }}
    if let s = obj as? CustomStringConvertible {{ return s.description }}
    return "unknown:\\(type(of: obj))"
}}
do {{
    let _ffl_seed = String(describing: {_bv}).count
    let _ffl_obj: AnyObject = _ffl_seed % 2 == 0
        ? _FflDerived() as AnyObject
        : _FflBase()    as AnyObject
    _ = _ffl_p5_cast(_ffl_obj)
    // Force-cast through Any — stresses value-witness metadata path
    let _ffl_any: Any = {_bv}
    _ = _ffl_any as? Int
    _ = _ffl_any as? String
    _ = _ffl_any as? Bool
    _ = type(of: _ffl_any)
}}
"""

        # --- P6: Non-copyable (~Copyable) ownership & consume/borrow ---
        # Targets: the move-only type verifier, consume/borrow operator
        # lowering, and the SIL ownership SSA verifier.
        # Move-only types must never be copied — the compiler must insert
        # explicit consumes and verify no path aliases a consumed value.
        p6 = f"""
// P6: Non-copyable (~Copyable) consume/borrow stress
struct _FflMoveOnly: ~Copyable {{
    var payload: Int
    init(_ v: Int) {{ payload = v }}
    consuming func consume() -> Int {{ payload }}
    borrowing func inspect() -> Int {{ payload }}
}}
func _ffl_p6_transfer(_ v: consuming _FflMoveOnly) -> Int {{
    v.consume()
}}
do {{
    var _ffl_mo = _FflMoveOnly(String(describing: {_bv}).count)
    _ = _ffl_mo.inspect()           // borrow — ownership retained
    let _ffl_result = _ffl_p6_transfer(_ffl_mo) // consume — ownership transferred
    _ = _ffl_result
    // Re-init after consume
    _ffl_mo = _FflMoveOnly(0)
    _ = _ffl_mo.inspect()
}}
"""

        # --- P7: Actor isolation & Sendable conformance ---
        # Targets: actor isolation checker, @MainActor, Sendable inference,
        # and the concurrency diagnostics pass in the Swift compiler.
        # Crossing actor isolation boundaries with a bridge value stresses
        # the data-race safety analysis without requiring actual concurrency.
        p7 = f"""
// P7: Actor isolation / Sendable / @MainActor stress
actor _FflCounter {{
    var count: Int = 0
    func increment(by n: Int) {{ count += n }}
    func get() -> Int {{ count }}
}}
@MainActor
func _ffl_p7_main_work(_ v: Int) -> String {{
    return "main:\\(v)"
}}
struct _FflSendableVal: Sendable {{
    let data: Int
}}
func _ffl_p7_drive() async {{
    let _ffl_actor = _FflCounter()
    let _ffl_n     = String(describing: {_bv}).count
    await _ffl_actor.increment(by: _ffl_n)
    let _ffl_c = await _ffl_actor.get()
    let _ffl_sv = _FflSendableVal(data: _ffl_c)
    _ = _ffl_sv
}}
"""

        # --- P8: Property wrapper composition & synthesised members ---
        # Targets: property wrapper type-checking, synthesised _storage
        # access paths, init(wrappedValue:) overload resolution, and
        # the SIL lowering of composed @propertyWrapper chains.
        p8 = f"""
// P8: Property wrapper composition stress
@propertyWrapper
struct _FflClamped<T: Comparable> {{
    private var _v: T
    let lo: T, hi: T
    init(wrappedValue: T, lo: T, hi: T) {{
        _v = min(max(wrappedValue, lo), hi)
        self.lo = lo; self.hi = hi
    }}
    var wrappedValue: T {{
        get {{ _v }}
        set {{ _v = min(max(newValue, lo), hi) }}
    }}
    var projectedValue: (T, T) {{ (lo, hi) }}
}}
@propertyWrapper
struct _FflLogged<T: CustomStringConvertible> {{
    var wrappedValue: T
    init(wrappedValue: T) {{ self.wrappedValue = wrappedValue }}
    var projectedValue: String {{ "logged:\\(wrappedValue)" }}
}}
struct _FflSettings {{
    @_FflClamped(lo: 0, hi: 100) var volume: Int = 50
    @_FflLogged var name: String = "default"
}}
do {{
    var _ffl_s = _FflSettings()
    _ffl_s.volume = String(describing: {_bv}).count % 200  // may exceed hi → clamped
    _ffl_s.name   = String(describing: {_bv})
    _ = _ffl_s.$name
    _ = _ffl_s.$volume
}}
"""

        # --- P9: Result builder & control-flow desugaring ---
        # Targets: @resultBuilder transform, buildBlock/buildOptional/
        # buildEither overload resolution, and the SIL lowering of
        # builder-transformed closures with complex control flow.
        p9 = f"""
// P9: @resultBuilder control-flow desugaring stress
@resultBuilder
struct _FflHTML {{
    static func buildBlock(_ parts: String...) -> String {{ parts.joined() }}
    static func buildOptional(_ part: String?) -> String {{ part ?? "" }}
    static func buildEither(first:  String) -> String {{ "<first>\\(first)</first>" }}
    static func buildEither(second: String) -> String {{ "<second>\\(second)</second>" }}
    static func buildArray(_ parts: [String]) -> String {{ parts.joined(separator: "\\n") }}
}}
func _ffl_p9_render(_ flag: Bool, items: [String]) -> String {{
    @_FflHTML var body: String {{
        "<root>"
        if flag {{
            "<active/>"
        }} else {{
            "<inactive/>"
        }}
        for item in items {{
            "<item>\\(item)</item>"
        }}
        if items.isEmpty {{
            "<empty/>"
        }}
        "</root>"
    }}
    return body
}}
do {{
    let _ffl_desc  = String(describing: {_bv})
    let _ffl_items = _ffl_desc.split(separator: " ").map(String.init)
    _ = _ffl_p9_render(_ffl_items.isEmpty, items: _ffl_items)
}}
"""

        return [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        sa = parent_a.content
        sb = parent_b.content

        # 1. Separate Structure
        a_body, a_imports = self._split_imports_and_body(sa)
        b_body, b_imports = self._split_imports_and_body(sb)
        all_imports = sorted(list(set(a_imports + b_imports)))

        # 2. Analyze Variables in A
        a_vars = self._extract_vars(a_body)

        # 3. Create Bridge
        # Pick a variable from A to inject into B
        bridge_code = ""
        bridge_var = None
        bridge_type = "Any"

        if a_vars:
            # Pick one
            v_name, v_type = random.choice(a_vars)

            # Define a bridge alias to avoid naming conflicts if 'v_name' is reused
            self.bridge_var_name = f"fusion_{uuid.uuid4().hex[:4]}"
            bridge_code = f"\n// --- Fusion Bridge ---\nvar {self.bridge_var_name} = {v_name}\n"

            bridge_var = self.bridge_var_name
            bridge_type = v_type

        # 4. Inject into B
        # Under --dataflow-type-match, prefer substituting a same-type B
        # variable's occurrences over the default literal-matching bridge.
        final_b_body = b_body
        if bridge_var:
            target_name = None
            if self.type_match:
                b_vars = self._extract_vars(b_body)
                target_name = self._pick_bridge_target_var(bridge_type, b_vars)

            if target_name:
                final_b_body = replace_word_occurrences(b_body, target_name, bridge_var)
            else:
                # Try to replace a compatible literal in B with the bridge variable
                modified_b, success = self._replace_compatible_literal(b_body, bridge_var, bridge_type)
                if success:
                    final_b_body = modified_b
                else:
                    # If no replacement possible, append a dummy usage to ensure validity
                    final_b_body += f"\nprint({bridge_var})\n"

        # 5. Pick one bug primitive and append it
        # Use a sentinel var if no bridge was created so primitives still compile
        prim_var  = bridge_var or "_ffl_sentinel"
        prim_type = bridge_type
        sentinel_decl = "" if bridge_var else "var _ffl_sentinel: Int = 0\n"

        primitives  = self._bug_primitives(prim_var, prim_type)
        chosen_prim = random.choice(primitives)

        # 6. Assemble: Imports -> A -> Bridge -> B -> Primitive
        # Note: A's variables are visible to Bridge and B because they are top-level
        final_content = "\n".join(all_imports) + "\n\n"
        final_content += f"// --- Seed A ---\n{a_body}\n"
        final_content += f"{bridge_code}\n"
        final_content += f"// --- Seed B ---\n{final_b_body}\n"
        final_content += f"{sentinel_decl}"
        final_content += f"// --- Bug Primitive ---\n{chosen_prim}\n"

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "swift",
                "description": f"Fused {parent_a.id} + {parent_b.id}"
            }
        )


class SwiftStateFusionStrategy(SwiftFusionStrategy):
    """
    State fusion for Swift (core/state_analysis.py): grafts one seed's
    continuation into the other's state at a profiled most complex state
    (`deinit`/`.close()`, an `as`/`as!`/`as?` cast, a `do`/`catch`
    boundary) instead of SwiftFusionStrategy's single-literal bridge +
    phase-directed bug primitive. Complements it rather than replacing it.
    """

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint, truncate_to_balanced

        host_src, donor_src = host.content, donor.content

        host_body, host_imports = self._split_imports_and_body(host_src)
        donor_body, donor_imports = self._split_imports_and_body(donor_src)
        all_imports = sorted(set(host_imports) | set(donor_imports))

        host_point = pick_state_point(host_body, "swift", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_body, "swift", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_body.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        donor_lines = donor_body.splitlines()
        start_idx = (donor_point.line_idx + 1) if donor_point else 0
        end_idx = truncate_to_balanced(donor_body, start_idx, "swift")
        truncated_donor = "\n".join(donor_lines[:end_idx])

        fused_body = graft_continuation(host_body, truncated_donor, host_point, donor_point)
        final_content = "\n".join(all_imports) + "\n" + fused_body

        return Seed(
            content=final_content,
            metadata={
                "parents": [host.id, donor.id],
                "type": "swift",
                "mode": f"state_{host_point.category}_{direction}",
                "description": f"State-fused {host.id} <- {donor.id} ({direction})",
            }
        )

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


class SwiftDeclarationFusionStrategy(SwiftFusionStrategy):
    """
    Declaration fusion for Swift: injects a donor-declared type/protocol
    into a host type's inheritance/conformance clause
    (`class Foo: Bar, DonorProto { ... }`). Swift resolves top-level
    declarations across the whole file in a first pass before checking
    bodies, so declaration order doesn't matter the way it does in
    PHP/Python — but a class/struct/enum claiming to conform to a
    protocol it doesn't fully implement, or a class trying to inherit
    from a struct/enum/protocol (only classes support superclass
    inheritance), is a pure declare/typecheck-time error, no call needed.
    """

    _SWIFT_TYPE_HEADER_RE = re.compile(
        r'\b(?P<kind>class|struct|enum|protocol)\s+(?P<name>[A-Za-z_]\w*)'
        r'(?:\s*<[^>]*>)?'
        r'(?P<conforms>\s*:\s*[^{]+)?'
        r'\s*\{'
    )

    def _inject_conformance(self, code: str, donor_name: str):
        matches = list(self._SWIFT_TYPE_HEADER_RE.finditer(code))
        if not matches:
            return code, False
        m = random.choice(matches)
        brace_pos = m.end() - 1
        if m.group('conforms'):
            new_code = code[:brace_pos].rstrip() + f", {donor_name} " + code[brace_pos:]
        else:
            new_code = code[:brace_pos].rstrip() + f": {donor_name} " + code[brace_pos:]
        return new_code, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_src, donor_src = host.content, donor.content
        host_body, host_imports = self._split_imports_and_body(host_src)
        donor_body, donor_imports = self._split_imports_and_body(donor_src)
        all_imports = sorted(set(host_imports) | set(donor_imports))

        donor_names = [m.group('name') for m in self._SWIFT_TYPE_HEADER_RE.finditer(donor_body)]
        fused_host, applied = host_body, False
        if donor_names:
            donor_name = random.choice(donor_names)
            fused_host, applied = self._inject_conformance(host_body, donor_name)

        final_content = "\n".join(all_imports) + "\n\n" + donor_body + "\n" + fused_host

        return Seed(content=final_content, metadata={
            "parents": [host.id, donor.id],
            "type": "swift",
            "mode": f"decl_{'conformance' if applied else 'none'}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


# ==========================================
# MLIR Specific Fusion Strategy
# ==========================================

class MLIRFusionStrategy(FusionStrategy):
    def __init__(self, project_root="projects/mlir", type_match: bool = False,
                 lightweight: bool = False):
        self.project_root = project_root
        # --dataflow-type-match: 90% of the time, require the bridge's
        # constants to share a scalar type (falling back to any two
        # constants — a deliberate, likely ill-typed pairing — when no
        # same-type group of >=2 exists), 10% of the time skip the
        # same-type search outright. Off by default — behavior is
        # unchanged from before this flag existed (always require same
        # scalar type, or emit no bridge at all).
        self.type_match = type_match
        # Dataflow fusion here is already on-the-fly (mlir_extract_constants
        # scans the seed directly, no cached metadata), so this is a no-op
        # for fuse() — it only exists so MLIRStateFusionStrategy (which
        # inherits this __init__) can read self.lightweight.
        self.lightweight = lightweight

    def _bug_primitives(self):
        """
        Phase-directed bug primitives for the MLIR compiler pipeline.
        Each primitive is a self-contained func.func targeting a distinct
        error-prone area: integer overflow, type narrowing, structured control
        flow nesting, memref aliasing, vector lowering, float edge cases,
        index boundary arithmetic, and multi-result call lowering.
        All symbols are prefixed _ffl_ to avoid collisions with seed code.
        Primitives are appended inside the wrapping module block.
        """

        # P1: Integer narrowing and extension chains — stresses arith lowering
        # and the folding patterns that truncate-then-extend constant values.
        p1 = '''
  // P1: Integer overflow, narrowing, and sign-extension chain
  func.func @_ffl_p1_overflow() -> i32 {
    %c_max = arith.constant 2147483647 : i32
    %c_one = arith.constant 1 : i32
    // addi wraps on two's-complement overflow — canonical undefined behaviour bait
    %wrapped = arith.addi %c_max, %c_one : i32
    // Widen to i64 then trunci back — stresses constant folding of trunci(extsi(x))
    %wide  = arith.extsi %wrapped : i32 to i64
    %c_big = arith.constant 65537 : i64
    %wide2 = arith.addi %wide, %c_big : i64
    %narrow = arith.trunci %wide2 : i64 to i16
    %ext   = arith.extsi %narrow : i16 to i32
    return %ext : i32
  }'''

        # P2: SCF for/if nesting — stresses lowering of iter_args through
        # nested structured regions (common source of dominance bugs).
        # Note: scf.for bounds/step must be `index`; carried values can be i64.
        p2 = '''
  // P2: scf.for with iter_args nested inside scf.if — dominance stress
  func.func @_ffl_p2_scf_nested() -> i64 {
    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %c16  = arith.constant 16 : index
    %acc0 = arith.constant 0 : i64
    %one  = arith.constant 1 : i64
    %two  = arith.constant 2 : i64
    %result = scf.for %iv = %c0 to %c16 step %c1
              iter_args(%acc = %acc0) -> (i64) {
      // Cast induction var to i64 for arithmetic
      %iv64 = arith.index_cast %iv : index to i64
      %odd  = arith.remui %iv64, %two : i64
      %cond = arith.cmpi eq, %odd, %acc0 : i64
      %next = scf.if %cond -> (i64) {
        %v = arith.muli %acc, %one : i64
        scf.yield %v : i64
      } else {
        %v = arith.addi %acc, %iv64 : i64
        scf.yield %v : i64
      }
      scf.yield %next : i64
    }
    return %result : i64
  }'''

        # P3: Memref dynamic alloc + store/load + dealloc — stresses the
        # bufferization pipeline and alias analysis when shape is unknown.
        p3 = '''
  // P3: Dynamic memref alloc/store/load — bufferization & alias stress
  func.func @_ffl_p3_memref() -> i32 {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c3  = arith.constant 3 : index
    %c4  = arith.constant 4 : index
    %mem = memref.alloc(%c4) : memref<?xi32>
    // Sequential stores: alias analysis must not reorder these
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 2147483647 : i32
    %v2 = arith.constant -2147483648 : i32
    %v3 = arith.constant -1 : i32
    memref.store %v0, %mem[%c0] : memref<?xi32>
    memref.store %v1, %mem[%c1] : memref<?xi32>
    memref.store %v2, %mem[%c3] : memref<?xi32>
    memref.store %v3, %mem[%c0] : memref<?xi32>   // overwrites index 0
    %loaded = memref.load %mem[%c0] : memref<?xi32>
    memref.dealloc %mem : memref<?xi32>
    return %loaded : i32
  }'''

        # P4: Vector broadcast + reduction — stresses VectorToLLVM lowering,
        # particularly around poison-value propagation and reduction identity.
        # Note: vector.splat was removed in LLVM 23; use vector.broadcast instead.
        p4 = '''
  // P4: Vector broadcast, arithmetic, reduction — VectorToLLVM stress
  func.func @_ffl_p4_vector() -> i32 {
    %c5    = arith.constant 5 : i32
    %cneg  = arith.constant -1 : i32
    %splat = vector.broadcast %c5   : i32 to vector<8xi32>
    %neg   = vector.broadcast %cneg : i32 to vector<8xi32>
    // Element-wise multiply — stresses vector element type lowering
    %prod = arith.muli %splat, %neg : vector<8xi32>
    // Horizontal add reduction with explicit neutral element
    %c0   = arith.constant 0 : i32
    %sum  = vector.reduction <add>, %prod, %c0 : vector<8xi32> into i32
    return %sum : i32
  }'''

        # P5: Float conversion roundtrip + NaN/Inf edge cases — stresses
        # arith constant folding for non-finite IEEE 754 values.
        p5 = '''
  // P5: Float conversion & NaN/Inf constant folding
  func.func @_ffl_p5_float() -> f64 {
    %imin = arith.constant -2147483648 : i32    // INT_MIN
    %f32  = arith.sitofp %imin : i32 to f32
    %f64  = arith.extf %f32 : f32 to f64
    // 0.0 / 0.0 produces NaN — constant folder must not crash
    %zero = arith.constant 0.0 : f64
    %nan  = arith.divf %zero, %zero : f64
    // minimumf with NaN: result must be NaN per IEEE 754-2019
    %r    = arith.minimumf %f64, %nan : f64
    // extf then truncf roundtrip: must be idempotent for finite values
    %back = arith.truncf %r : f64 to f32
    %out  = arith.extf %back : f32 to f64
    return %out : f64
  }'''

        # P6: Index/i64 interop at integer boundary — stresses index_cast
        # when the value saturates the platform word size.
        p6 = '''
  // P6: Index boundary arithmetic — index_cast at i64 max
  func.func @_ffl_p6_index_boundary() -> index {
    %large = arith.constant 9223372036854775807 : i64   // i64 MAX
    %idx   = arith.index_cast %large : i64 to index
    %c1    = arith.constant 1 : index
    // Adding 1 to max index — undefined on 32-bit targets, wraps on 64-bit
    %r     = arith.addi %idx, %c1 : index
    // Cast back and verify round-trip via i32 (lossy, stresses trunci)
    %i64   = arith.index_cast %r : index to i64
    %i32   = arith.trunci %i64 : i64 to i32
    %back  = arith.index_cast %i32 : i32 to index
    return %back : index
  }'''

        # P7: scf.while — stresses the do-while lowering to CFG and the
        # "before"/"after" region dominance requirements.
        p7 = '''
  // P7: scf.while loop — structured do-while lowering stress
  func.func @_ffl_p7_while() -> i32 {
    %c0  = arith.constant 0 : i32
    %c1  = arith.constant 1 : i32
    %c10 = arith.constant 10 : i32
    %res = scf.while (%arg = %c0) : (i32) -> i32 {
      // before region: compute condition
      %cond = arith.cmpi slt, %arg, %c10 : i32
      scf.condition(%cond) %arg : i32
    } do {
    ^bb0(%arg : i32):
      // after region: advance loop variable
      %next = arith.addi %arg, %c1 : i32
      scf.yield %next : i32
    }
    return %res : i32
  }'''

        # P8: Bitwise boundary operations — stresses arith lowering for
        # shifts with large shift amounts (shift >= bitwidth is UB in LLVM IR).
        p8 = '''
  // P8: Bitwise operations at integer boundaries — shift UB stress
  func.func @_ffl_p8_bitwise() -> i64 {
    %allones = arith.constant -1 : i64            // 0xFFFFFFFFFFFFFFFF
    %min64   = arith.constant -9223372036854775808 : i64
    %c63     = arith.constant 63 : i64
    // Arithmetic right-shift of MIN by 63 — all bits become sign bit
    %shr  = arith.shrsi %min64, %c63 : i64
    // Left-shift all-ones by 0 — identity, should fold
    %c0   = arith.constant 0 : i64
    %shl  = arith.shli %allones, %c0 : i64
    %xor  = arith.xori %shr, %shl : i64
    %and  = arith.andi %xor, %min64 : i64
    %or   = arith.ori  %and, %allones : i64
    return %or : i64
  }'''

        # P9: Multi-result func.call — stresses multi-value SSA lowering and
        # the ABI expansion of functions returning more than one scalar.
        p9 = '''
  // P9: Multi-result function & call — multi-value SSA lowering
  func.func @_ffl_p9_divmod(%a : i32, %b : i32) -> (i32, i32) {
    %q = arith.divsi %a, %b : i32
    %r = arith.remsi %a, %b : i32
    return %q, %r : i32, i32
  }

  func.func @_ffl_p9_call() -> i32 {
    %num = arith.constant 1000000007 : i32
    %den = arith.constant 998244353 : i32
    %q, %r = func.call @_ffl_p9_divmod(%num, %den) : (i32, i32) -> (i32, i32)
    %res = arith.addi %q, %r : i32
    return %res : i32
  }'''

        all_phases = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
        selected = random.choice(all_phases)
        return selected

    def _make_bridge(self, a_consts, b_consts, uid):
        """
        Build a standalone bridge function that combines constants extracted
        from both seeds.  Defined at module scope — no injection into any
        existing function body (which was fragile and caused cross-module
        reference errors when mlir_strip_outer_module failed).
        Returns '' when there are no constants to work with.

        --dataflow-type-match (self.type_match): 90% of the time, require a
        same-scalar-type group of >=2 constants as before, falling back to
        a type-agnostic pool when none exists instead of giving up (no
        bridge); 10% of the time, skip the same-type search outright and
        draw from all constants regardless of type — deliberately risking
        a constant declared with another constant's (wrong) type, the same
        tolerance for likely-invalid cross-seed values that Clang's
        dataflow fusion already applies. With the flag off, behavior is
        unchanged: only ever bridge same-type constants, or emit no bridge.
        """
        all_consts = a_consts + b_consts
        if not all_consts:
            return ""

        # Try to find a group of >=2 constants sharing a simple scalar type.
        ty = None
        same = []
        for candidate_ty in dict.fromkeys(c["ty"] for c in all_consts):
            group = [c for c in all_consts if c["ty"] == candidate_ty]
            # Only use simple scalar types to stay safe across all passes
            if len(group) >= 2 and re.fullmatch(r'[iuf]\d+|index', candidate_ty):
                ty = candidate_ty
                same = group
                break

        if self.type_match and (not same or random.random() >= 0.90):
            same = []  # force the type-agnostic fallback below

        if same:
            n = min(pick_occurrence_count(), len(same))
            picks = random.sample(same, n)
        elif self.type_match:
            pool = [c for c in all_consts if re.fullmatch(r'[iuf]\d+|index', c["ty"])] or all_consts
            n = min(pick_occurrence_count(), len(pool))
            picks = random.sample(pool, n)
            ty = picks[0]["ty"]
        else:
            return ""

        op = "arith.addf" if ty.startswith("f") else "arith.addi"

        lines = [f"func.func @_ffl_bridge_{uid}() -> {ty} {{"]
        varnames = []
        for i, p in enumerate(picks):
            vn = f"%_b{i}"
            lines.append(f"  {vn} = arith.constant {p['lit']} : {ty}")
            varnames.append(vn)

        acc = varnames[0]
        for vn in varnames[1:]:
            nacc = f"%_bacc_{vn[1:]}"
            lines.append(f"  {nacc} = {op} {acc}, {vn} : {ty}")
            acc = nacc

        lines.append(f"  return {acc} : {ty}")
        lines.append("}")
        return "\n".join(lines)

    # Patterns that indicate obviously non-MLIR content (LLM pseudo-code or old std dialect).
    _NON_MLIR_PATTERNS = [
        re.compile(r':\s*string\b'),                  # `string` is not an MLIR type
        re.compile(r'(?<!["\w])==(?![>="\w])'),       # bare == operator (Python/Java)
        re.compile(r'"std\.'),                        # ancient std dialect string ops
        re.compile(r'\bstd\.(?:constant|addi|load|store|call|return)\b'),  # old std dialect ops
        re.compile(r'\bimport\s+\w'),                 # Python/Java import statement
        re.compile(r'\bdef\s+\w+\s*\('),              # Python def
        re.compile(r'\bclass\s+\w+'),                 # OOP class keyword
        # Old std dialect ops used without dialect prefix (very common in LLM seeds)
        re.compile(r'=\s*constant\s+[\d"(+-]'),       # bare `constant` (no arith. prefix)
        re.compile(r'=\s*(?:addi|subi|muli|divi|addf|subf|mulf|divf)\s+%'),  # bare arith ops
        re.compile(r'=\s*(?:cmpi|cmpf)\s+\w'),       # bare comparison ops
        re.compile(r'=\s*(?:alloc|store|load)\s*[(%]'),  # bare memref ops
        re.compile(r'%\w+\s*=\s*type\s*\{'),          # LLVM IR type definition
        re.compile(r'\balloca\b(?!\s+[^,]*memref)'),  # bare alloca (LLVM IR)
        # Invalid arith ops (LLM hallucinations)
        re.compile(r'\barith\.divi\b'),               # divi doesn't exist (use divsi/divui)
        re.compile(r'\barith\.modi\b'),               # modi doesn't exist (use remsi/remui)
        re.compile(r'\bfunc\.constant\b'),            # func.constant doesn't exist
        # Bare control flow (LLM pseudo-code)
        re.compile(r'(?<!\w)for\s+%\w+\s+in\b'),     # Python-style `for %x in`
        re.compile(r'(?<!\w)if\s+%\w+\s*:'),         # Python-style `if %x:`
        # String literals used as SSA values: `%x = "some string" : !type`
        # Valid MLIR uses `"op.name"(args)` — without `(` it's a string value, not an op.
        re.compile(r'=\s*"[^"]*"\s*:\s*!'),          # bare string literal as SSA value
        # Invalid MLIR types that never exist in any registered dialect
        re.compile(r':\s*!(?:string|void|llvm\.str(?:ing)?|object)\b'),
    ]
    _HAS_MLIR_STRUCTURE = re.compile(
        r'func\.func\b|arith\.\w|scf\.\w|memref\.\w|module\s*\{|cf\.\w|linalg\.\w'
    )

    def _is_plausible_mlir(self, body: str) -> bool:
        """Return False if body looks like LLM-generated pseudo-code rather than MLIR."""
        for pat in self._NON_MLIR_PATTERNS:
            if pat.search(body):
                return False
        if not self._HAS_MLIR_STRUCTURE.search(body):
            return False
        return True

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        a_src = parent_a.content
        b_src = parent_b.content
        a_id = parent_a.id
        b_id = parent_b.id

        # 1) Strip FileCheck / lit directives FIRST.
        #    Most MLIR test seeds start with '// RUN:' / '// CHECK:' lines.
        #    Leaving them in breaks mlir_strip_outer_module (it bails when
        #    'module' is not at position 0), which causes A/B to keep their
        #    own 'module {}' wrappers.
        a_src = mlir_strip_directives(a_src)
        b_src = mlir_strip_directives(b_src)

        # 1b) Upgrade bare `func @name` (LLVM <23) to `func.func @name`.
        a_src = _MLIR_BARE_FUNC_RE.sub('func.func @', a_src)
        b_src = _MLIR_BARE_FUNC_RE.sub('func.func @', b_src)

        # 2) Rename symbols to avoid collisions
        a_ren = mlir_rename_symbols(a_src, f"A_{a_id}_")
        b_ren = mlir_rename_symbols(b_src, f"B_{b_id}_")

        # 3) Strip outer module wrappers (now reliably works after directive strip)
        a_body = mlir_strip_outer_module(a_ren).strip()
        b_body = mlir_strip_outer_module(b_ren).strip()

        # 3b) Remove internal // ----- separators from bodies: these would be
        #     treated by --split-input-file as section boundaries, splitting the
        #     body mid-way and exposing fragments that are never valid on their own.
        a_body = re.sub(r'^\s*//\s*-{3,}.*$', '', a_body, flags=re.MULTILINE).strip()
        b_body = re.sub(r'^\s*//\s*-{3,}.*$', '', b_body, flags=re.MULTILINE).strip()

        # 3c) Replace bodies that are clearly not valid MLIR (LLM pseudo-code)
        #     with an empty comment so that section parses cleanly.
        if not self._is_plausible_mlir(a_body):
            a_body = "// (seed A not plausible MLIR — omitted)"
        if not self._is_plausible_mlir(b_body):
            b_body = "// (seed B not plausible MLIR — omitted)"

        # 4) Build standalone bridge from constants in both bodies.
        a_consts = mlir_extract_constants(a_body)
        b_consts = mlir_extract_constants(b_body)
        uid = f"{a_id}_{b_id}"
        bridge_code = self._make_bridge(a_consts, b_consts, uid)

        # 5) Bug primitives
        bug_prims = self._bug_primitives()

        def _indent(text):
            return "\n".join("  " + ln if ln.strip() else "" for ln in text.splitlines())

        def _hoist_aliases(body: str):
            """
            Split file-level attribute/type alias lines from module-level content.
            Alias lines start with '#' or '!' at the beginning (after stripping).
            Returns (aliases_block, remaining_body).
            """
            alias_lines, body_lines = [], []
            in_aliases = True
            for line in body.splitlines():
                stripped = line.strip()
                if in_aliases and (stripped.startswith('#') or stripped.startswith('!')):
                    alias_lines.append(line)
                else:
                    in_aliases = False
                    body_lines.append(line)
            return "\n".join(alias_lines), "\n".join(body_lines).strip()

        def _wrap_section(body: str, section_comment: str):
            """Return lines for one // ----- section, handling alias hoisting and
            bodies that already contain a top-level module {} wrapper."""
            aliases, inner = _hoist_aliases(body)
            parts = [section_comment]
            if aliases:
                parts.append(aliases)
            # If inner already starts with 'module' it was not stripped — emit directly.
            inner_stripped = inner.lstrip()
            if inner_stripped.startswith("module"):
                parts.append(inner)
            else:
                parts.append("module {")
                parts.append(_indent(inner))
                parts.append("}")
            return "\n".join(parts)

        # Output three independent modules separated by // ----- so that
        # --split-input-file evaluates each section on its own.  If seed A is
        # individually valid and seed B is individually valid, the fused file
        # passes as a whole, dramatically increasing the zero-rate.
        # The bridge + bug-primitives section is always syntactically correct
        # (self-contained arith ops) so it never pulls the return code to 1.
        fused_parts = [
            f"// FUSED MLIR (FFL)  A: {a_id}  B: {b_id}",
            "",
            _wrap_section(a_body, "// ===== Section A ====="),
            "",
            "// -----",
            "",
            _wrap_section(b_body, "// ===== Section B ====="),
            "",
            "// -----",
            "",
            "// ===== FFL Bridge + Bug Primitives =====",
            "module {",
        ]
        if bridge_code:
            fused_parts.append(_indent(bridge_code))
            fused_parts.append("")
        fused_parts.append(bug_prims)
        fused_parts.append("}")

        final_code = "\n".join(fused_parts) + "\n"

        return Seed(
            content=final_code,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "mlir",
                "description": f"Fused {parent_a.id} + {parent_b.id}"
            }
        )


class MLIRStateFusionStrategy(MLIRFusionStrategy):
    """
    State fusion for MLIR (core/state_analysis.py). Unlike
    MLIRFusionStrategy's default mode — which emits seed A and seed B as
    two independent `--split-input-file` sections plus a separate bridge
    module, so each half stands alone — this strategy actually grafts
    donor ops into the host's op stream at a profiled most-complex-state
    point (right after a `memref.dealloc`, a cast op, or a `cf.assert`),
    inside one shared `module { }`, so the verifier has to resolve the
    donor's ops (SSA operands, types) in the host's context rather than
    in isolation. "most complex state" is an approximation for a
    declarative IR (see core/state_analysis.py's mlir entry) — this is
    the closest analogue to the paper's runtime-state grafting available
    without an execution model.
    """

    # MLIR blocks require exactly one terminator op, and it must be the
    # last op in the block — unlike brace balance, this isn't something
    # truncate_to_balanced's generic paren/brace counting can know about,
    # so a donor continuation must also be cut before its own terminator
    # to avoid producing a block with a terminator followed by more ops
    # (or two terminators) once grafted into the host's block.
    _MLIR_TERMINATOR_RE = re.compile(r'^\s*(?:func\.return|return|cf\.br|cf\.cond_br|scf\.yield)\b')

    @classmethod
    def _truncate_before_terminator(cls, lines: List[str], start_idx: int, end_idx: int) -> int:
        for i in range(start_idx, end_idx):
            if cls._MLIR_TERMINATOR_RE.match(lines[i]):
                return i
        return end_idx

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint, truncate_to_balanced

        host_src = mlir_strip_directives(host.content)
        donor_src = mlir_strip_directives(donor.content)
        host_src = _MLIR_BARE_FUNC_RE.sub('func.func @', host_src)
        donor_src = _MLIR_BARE_FUNC_RE.sub('func.func @', donor_src)

        host_ren = mlir_rename_symbols(host_src, f"H_{host.id}_")
        donor_ren = mlir_rename_symbols(donor_src, f"D_{donor.id}_")

        host_body = mlir_strip_outer_module(host_ren).strip()
        donor_body = mlir_strip_outer_module(donor_ren).strip()
        host_body = re.sub(r'^\s*//\s*-{3,}.*$', '', host_body, flags=re.MULTILINE).strip()
        donor_body = re.sub(r'^\s*//\s*-{3,}.*$', '', donor_body, flags=re.MULTILINE).strip()

        if not self._is_plausible_mlir(host_body) or not self._is_plausible_mlir(donor_body):
            # Not plausibly MLIR (e.g. LLM-authored pseudo-code seed) —
            # emit the host alone rather than grafting unparseable text.
            return Seed(content=f"module {{\n{host_body}\n}}\n", metadata={
                "parents": [host.id, donor.id], "type": "mlir",
                "mode": f"state_fallback_{direction}",
                "description": f"State-fused (fallback, non-plausible donor) {host.id} <- {donor.id}",
            })

        host_point = pick_state_point(host_body, "mlir", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_body, "mlir", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_body.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        donor_lines = donor_body.splitlines()
        start_idx = (donor_point.line_idx + 1) if donor_point else 0
        end_idx = truncate_to_balanced(donor_body, start_idx, "mlir")
        end_idx = self._truncate_before_terminator(donor_lines, start_idx, end_idx)
        truncated_donor = "\n".join(donor_lines[:end_idx])

        fused_body = graft_continuation(host_body, truncated_donor, host_point, donor_point)
        final_code = f"module {{\n{fused_body}\n}}\n"

        return Seed(content=final_code, metadata={
            "parents": [host.id, donor.id],
            "type": "mlir",
            "mode": f"state_{host_point.category}_{direction}",
            "description": f"State-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


class MLIRDeclarationFusionStrategy(MLIRFusionStrategy):
    """
    Declaration fusion for MLIR — the paper's "operand type constraints"
    instance. Swaps one declared type in a host `func.func` signature
    (an argument type or the result type) for a type token taken from a
    donor signature, while leaving the host's body untouched. The body
    still produces/consumes the *original* type, so the signature no
    longer matches what the body actually does — a pure declaration-level
    mismatch the verifier catches at parse/verify time, requiring no
    execution (MLIR is never executed by this fuzzer at all, only
    verified/compiled, so the whole domain leans declaration-fusion — see
    class docstring on MLIRStateFusionStrategy for the state-fusion
    counterpart, which grafts ops instead of swapping a type).
    """

    _MLIR_TYPE_TOKEN_RE = re.compile(
        r'\bi\d+\b|\bf(?:16|32|64|80|128)\b|\bindex\b'
        r'|\bmemref<[^>]*>|\btensor<[^>]*>|\bvector<[^>]*>'
    )
    _MLIR_FUNC_SIG_RE = re.compile(
        r'func\.func\s+@(?P<name>[A-Za-z_][\w.$-]*)\s*\((?P<args>[^)]*)\)'
        r'\s*(?:->\s*(?P<ret>[^\{]+?))?\s*\{'
    )

    def _donor_type_tokens(self, body: str):
        tokens = set()
        for m in self._MLIR_FUNC_SIG_RE.finditer(body):
            seg = (m.group('args') or '') + ' ' + (m.group('ret') or '')
            tokens.update(self._MLIR_TYPE_TOKEN_RE.findall(seg))
        return sorted(tokens)

    def _inject_type_swap(self, body: str, donor_type: str):
        matches = list(self._MLIR_FUNC_SIG_RE.finditer(body))
        if not matches:
            return body, False
        m = random.choice(matches)
        seg_start, seg_end = m.start(), m.end()
        segment = body[seg_start:seg_end]
        type_matches = list(self._MLIR_TYPE_TOKEN_RE.finditer(segment))
        if not type_matches:
            return body, False
        tm = random.choice(type_matches)
        new_segment = segment[:tm.start()] + donor_type + segment[tm.end():]
        new_body = body[:seg_start] + new_segment + body[seg_end:]
        return new_body, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_src = mlir_strip_directives(host.content)
        donor_src = mlir_strip_directives(donor.content)
        host_src = _MLIR_BARE_FUNC_RE.sub('func.func @', host_src)
        donor_src = _MLIR_BARE_FUNC_RE.sub('func.func @', donor_src)
        host_ren = mlir_rename_symbols(host_src, f"H_{host.id}_")
        donor_ren = mlir_rename_symbols(donor_src, f"D_{donor.id}_")
        host_body = mlir_strip_outer_module(host_ren).strip()
        donor_body = mlir_strip_outer_module(donor_ren).strip()
        host_body = re.sub(r'^\s*//\s*-{3,}.*$', '', host_body, flags=re.MULTILINE).strip()
        donor_body = re.sub(r'^\s*//\s*-{3,}.*$', '', donor_body, flags=re.MULTILINE).strip()

        if not self._is_plausible_mlir(host_body) or not self._is_plausible_mlir(donor_body):
            return Seed(content=f"module {{\n{host_body}\n}}\n", metadata={
                "parents": [host.id, donor.id], "type": "mlir",
                "mode": f"decl_fallback_{direction}",
                "description": f"Declaration-fused (fallback, non-plausible donor) {host.id} <- {donor.id}",
            })

        donor_types = self._donor_type_tokens(donor_body)
        fused_host, applied = host_body, False
        if donor_types:
            donor_type = random.choice(donor_types)
            fused_host, applied = self._inject_type_swap(host_body, donor_type)

        # Donor kept in the same module (not grafted into host — the
        # swap above is the whole point) so it's still a self-consistent
        # section the verifier can check independently.
        final_code = f"module {{\n{fused_host}\n\n{donor_body}\n}}\n"

        return Seed(content=final_code, metadata={
            "parents": [host.id, donor.id],
            "type": "mlir",
            "mode": f"decl_{'type_swap' if applied else 'none'}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


class RustLexMixin:
    """
    Literal-aware lexing helpers for Rust source text (strings, chars,
    comments, raw/byte strings), shared by every Rust fusion strategy that
    needs to find item/brace boundaries without a real parser.
    """

    _RUST_RAW_STR_START = re.compile(r'(?:b)?r(#*)"')
    _RUST_CHAR_LIT_RE = re.compile(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^'\\])'")

    def _rust_skip_literal(self, code: str, i: int):
        """If code[i:] starts a string/char/comment literal, return the
        index just past its end; else return None."""
        n = len(code)
        two = code[i:i + 2]
        if two == '//':
            j = code.find('\n', i)
            return j if j != -1 else n
        if two == '/*':
            j = code.find('*/', i + 2)
            return (j + 2) if j != -1 else n
        m = self._RUST_RAW_STR_START.match(code, i)
        if m:
            end_pat = '"' + m.group(1)
            j = code.find(end_pat, m.end())
            return (j + len(end_pat)) if j != -1 else n
        ch = code[i]
        if ch == '"' or two == 'b"':
            j = i + (2 if two == 'b"' else 1)
            while j < n:
                if code[j] == '\\':
                    j += 2
                    continue
                if code[j] == '"':
                    return j + 1
                j += 1
            return n
        if ch == "'":
            cm = self._RUST_CHAR_LIT_RE.match(code, i)
            if cm:
                return cm.end()
            # lifetime, e.g. 'a — no closing quote
            j = i + 1
            while j < n and (code[j].isalnum() or code[j] == '_'):
                j += 1
            return j
        return None

    def _rust_real_mask(self, code: str):
        """bool list: True where code[i] is real code, False inside a
        string/char/comment literal."""
        n = len(code)
        is_real = [False] * n
        i = 0
        while i < n:
            ch = code[i]
            prev = code[i - 1] if i > 0 else ''
            maybe_literal = (
                ch in ('"', "'") or code[i:i + 2] in ('//', '/*') or
                (ch in ('b', 'r') and not (prev.isalnum() or prev == '_'))
            )
            end = self._rust_skip_literal(code, i) if maybe_literal else None
            if end is not None:
                i = end
                continue
            is_real[i] = True
            i += 1
        return is_real

    def _rust_matching_close(self, code, is_real, open_pos, open_ch='{', close_ch='}'):
        """open_pos is the index of an open_ch; return the index of its
        matching close_ch, respecting nesting and literals."""
        depth = 0
        n = len(code)
        i = open_pos
        while i < n:
            if is_real[i]:
                if code[i] == open_ch:
                    depth += 1
                elif code[i] == close_ch:
                    depth -= 1
                    if depth == 0:
                        return i
            i += 1
        return None

    def _rust_split_top_level(self, text: str, sep: str = ','):
        """Split text on `sep` at bracket depth 0 (tracks ([{< nesting)."""
        is_real = self._rust_real_mask(text)
        parts, cur, depth = [], [], 0
        for i, ch in enumerate(text):
            if is_real[i]:
                if ch in '([{<':
                    depth += 1
                elif ch in ')]}>':
                    depth = max(0, depth - 1)
                elif ch == sep and depth == 0:
                    parts.append(''.join(cur))
                    cur = []
                    continue
            cur.append(ch)
        if ''.join(cur).strip():
            parts.append(''.join(cur))
        return [p for p in parts if p.strip()]


class RustFusionStrategy(RustLexMixin, FusionStrategy):
    """
    Rust-Specific Fusion Strategy.

    Unlike the old "call both mains back-to-back" approach, this strategy
    inlines eligible free-function calls directly into each seed's `main`
    body (substituting the real call-site arguments — not fabricated
    values), splits both inlined bodies into statements, and interleaves
    them into ONE shared scope. That's what lets a variable that used to
    live inside a helper `fn` actually interact with a variable from the
    other seed: they now sit in the same block.

    Steps:
    1. Extract crate-level attrs / 'use' imports and hoist them.
    2. Rename 'fn main' in each parent to a unique name.
    3. Find top-level (non-method, non-generic, non-recursive, `?`-free)
       free functions and inline their call sites inside `main`, rewriting
       `return expr;` to `break 'label expr;` inside a labeled block so
       control flow still works once the function body is flattened.
    4. Split each inlined `main` body into statements (order preserved per
       seed — Rust's move/borrow rules make reordering within a seed risky)
       and riffle-interleave the two statement streams into one function.
    5. Optionally inject one type-checked cross-seed bridge: pick a later
       `let` in one seed and rebind it from an earlier, type-compatible
       `let` in the other seed (Copy direct, Clone via `.clone()`).

    Functions this can't safely flatten (generics, recursion, `?`, or that
    aren't called from `main`) are left as ordinary top-level items in the
    output — unused is harmless, a compile error is not.
    """
    def __init__(self, project_root="projects/rust", type_match: bool = False):
        self.project_root = project_root
        self.mut = RustMutator()
        self.bridge_var_name = "fusion_var"
        # --dataflow-type-match: 90% of the time, prefer a type-compatible
        # cross-seed `let` (falling back to any earlier cross-seed `let`
        # when none is compatible), 10% of the time skip the type filter
        # outright. Off by default — behavior is unchanged from before this
        # flag existed (always require type compatibility, as _bridge_
        # across_seeds did originally).
        self.type_match = type_match

    def _process_seed(self, code, uid):
        """
        Parses seed code:
        - Extracts crate-level #![...] attributes and 'use' imports separately.
        - Renames 'main' -> 'main_<uid>'.
        """
        crate_attrs = []  # #![...] lines — must be at crate top
        use_lines = []    # use ... lines
        body_lines = []

        lines = code.splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#!"):
                crate_attrs.append(line)
            elif stripped.startswith("use ") or stripped.startswith("extern crate "):
                use_lines.append(line)
            else:
                body_lines.append(line)

        body = "\n".join(body_lines)

        # Corpus seed ids are often derived from filenames (e.g.
        # "aapcs-unwind.rs"), which contain hyphens/dots — invalid in Rust
        # identifiers. Sanitize before using uid in any generated name.
        uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid)

        # Rename main function
        main_regex = r'(fn\s+)main(\s*\()'
        if re.search(main_regex, body):
            new_main = f"main_{uid}"
            body = re.sub(main_regex, f"\\1{new_main}\\2", body, count=1)
        else:
            new_main = None

        return crate_attrs, use_lines, body, new_main

    # ------------------------------------------------------------------
    # Type-aware bridge helpers
    # ------------------------------------------------------------------

    # Primitive (Copy) types: bridge directly — no .clone() needed.
    _COPY_TYPES = frozenset({
        "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char",
    })
    # Numeric types — two numerics are always compatible.
    _NUMERIC_TYPES = frozenset({
        "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64",
    })

    def _base_type(self, type_str: str) -> str:
        """Strip &/&mut/lifetime prefixes and generic params to get the leaf name."""
        t = re.sub(r"^&(?:'[a-z_]+\s+)?(?:mut\s+)?", "", (type_str or "").strip())
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', t)
        return m.group(1) if m else t

    def _types_compatible(self, ta: str, tb: str) -> bool:
        """Return True if bridging a variable of type ta into a slot of type tb is safe."""
        ba, bb = self._base_type(ta), self._base_type(tb)
        if ba == bb:
            return True
        # Both numeric → compatible (may need an as-cast, but compiles)
        if ba in self._NUMERIC_TYPES and bb in self._NUMERIC_TYPES:
            return True
        return False

    def _replace_random_identifier(self, s: str, name: str, replacement: str):
        """Word-boundary-safe replace of a weighted-random number of
        occurrences (see pick_occurrence_count) of `name` in `s` (not
        `->`/`::`-qualified). Returns None if no match — callers must not
        fall back to plain substring replace, which would also match `name`
        as a substring of unrelated identifiers (e.g. `a` inside `break`)."""
        pattern = re.compile(r'(?<![.\w:])' + re.escape(name) + r'(?!\w)')
        matches = list(pattern.finditer(s))
        if not matches:
            return None
        n = min(pick_occurrence_count(), len(matches))
        chosen = sorted(random.sample(matches, n), key=lambda m: m.start(), reverse=True)
        for m in chosen:
            s = s[:m.start()] + replacement + s[m.end():]
        return s

    # ------------------------------------------------------------------
    # Top-level free-function extraction & inlining
    # ------------------------------------------------------------------

    _RUST_FN_START_RE = re.compile(
        r'(?:pub(?:\([^)]*\))?\s+)?'
        r'(?:(?:async|unsafe|const)\s+){0,3}'
        r'fn\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    )
    _RUST_PARAM_RE = re.compile(r'^\s*(?:mut\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$', re.S)
    _LET_TYPE_RE = re.compile(r'\blet\s+(?:mut\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=;\n]+?)\s*(?:=|;)')

    def _extract_top_level_fns(self, body: str):
        """Find `fn` items at brace-depth 0 (i.e. not inside impl/trait/mod
        — those are methods, not free functions, and are left alone)."""
        is_real = self._rust_real_mask(body)
        n = len(body)
        depth_before = [0] * n
        depth = 0
        for i in range(n):
            if is_real[i]:
                depth_before[i] = depth
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth = max(0, depth - 1)

        fns = []
        for m in self._RUST_FN_START_RE.finditer(body):
            start = m.start()
            if not is_real[start] or depth_before[start] != 0:
                continue
            name = m.group(1)
            is_unsafe = 'unsafe' in m.group(0)
            pos = m.end()
            while pos < n and body[pos].isspace():
                pos += 1
            if pos < n and body[pos] == '<':
                gdepth = 0
                while pos < n:
                    if is_real[pos]:
                        if body[pos] == '<':
                            gdepth += 1
                        elif body[pos] == '>':
                            gdepth -= 1
                            if gdepth == 0:
                                pos += 1
                                break
                    pos += 1
                generics_present = True
                while pos < n and body[pos].isspace():
                    pos += 1
            else:
                generics_present = False
            if pos >= n or body[pos] != '(':
                continue
            paren_close = self._rust_matching_close(body, is_real, pos, '(', ')')
            if paren_close is None:
                continue
            params_text = body[pos + 1:paren_close]
            pos = paren_close + 1
            bdepth = 0
            while pos < n:
                if is_real[pos]:
                    ch = body[pos]
                    if ch in '([<':
                        bdepth += 1
                    elif ch in ')]>':
                        bdepth = max(0, bdepth - 1)
                    elif ch == '{' and bdepth == 0:
                        break
                pos += 1
            if pos >= n or body[pos] != '{':
                continue
            body_close = self._rust_matching_close(body, is_real, pos, '{', '}')
            if body_close is None:
                continue
            fns.append({
                'name': name,
                'start': start,
                'end': body_close + 1,
                'params_text': params_text,
                'generics': generics_present,
                'is_unsafe': is_unsafe,
                'body_text': body[pos + 1:body_close],
            })
        return fns

    def _parse_params(self, params_text: str):
        """Return [(name, type), ...] or None if unsupported (self receiver
        or a destructuring pattern we can't safely bind by name)."""
        params = []
        for p in self._rust_split_top_level(params_text, ','):
            p = p.strip()
            if not p:
                continue
            if re.match(r'^&(?:\'[a-z_]+\s+)?(?:mut\s+)?self\b', p) or p == 'self':
                return None
            m = self._RUST_PARAM_RE.match(p)
            if not m:
                return None
            params.append((m.group(1), m.group(2).strip()))
        return params

    def _fn_is_inlinable(self, fn: dict, params) -> bool:
        if fn['generics'] or params is None:
            return False
        if re.search(r'\b' + re.escape(fn['name']) + r'\s*\(', fn['body_text']):
            return False  # (mutually) recursive-looking — skip for safety
        is_real = self._rust_real_mask(fn['body_text'])
        for i, ch in enumerate(fn['body_text']):
            if ch == '?' and is_real[i]:
                return False  # `?` operator needs the enclosing fn's Result/Option type
        return True

    def _flatten_fn_call(self, fn: dict, params, args_text: str, label: str) -> str:
        args = [a.strip() for a in self._rust_split_top_level(args_text, ',') if a.strip()]
        lets = [f"let {pname}: {ptype} = {arg};" for (pname, ptype), arg in zip(params, args)]
        body = fn['body_text']
        body = re.sub(r'\breturn\s*;', f"break '{label};", body)
        body = re.sub(r'\breturn\s+([^;]+);', rf"break '{label} \1;", body)
        inner = "\n".join(lets) + "\n" + body
        if fn['is_unsafe']:
            inner = f"unsafe {{\n{inner}\n}}"
        return f"'{label}: {{\n{inner}\n}}"

    def _inline_calls_in_body(self, main_body: str, fn_defs: list, uid: str) -> str:
        """Replace direct calls to any inlinable fn in fn_defs with its
        flattened labeled-block body, substituting the real call arguments."""
        uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid)
        counter = 0
        for fn in fn_defs:
            params = self._parse_params(fn['params_text'])
            if not self._fn_is_inlinable(fn, params):
                continue
            # Compute the literal mask once per function and locate every
            # call site up front, then splice back-to-front so earlier
            # offsets stay valid — avoids re-scanning the whole (growing)
            # body from scratch for every single occurrence.
            is_real = self._rust_real_mask(main_body)
            call_re = re.compile(r'\b' + re.escape(fn['name']) + r'\s*\(')
            spans = []
            for cand in call_re.finditer(main_body):
                if not is_real[cand.start()]:
                    continue
                open_paren = cand.end() - 1
                close_paren = self._rust_matching_close(main_body, is_real, open_paren, '(', ')')
                if close_paren is not None:
                    spans.append((cand.start(), open_paren, close_paren))
            for start, open_paren, close_paren in reversed(spans):
                args_text = main_body[open_paren + 1:close_paren]
                counter += 1
                label = f"ffl_{uid}_{fn['name']}_{counter}"
                replacement = self._flatten_fn_call(fn, params, args_text, label)
                main_body = main_body[:start] + replacement + main_body[close_paren + 1:]
        return main_body

    def _flatten_and_extract_main(self, body: str, main_name, uid: str):
        """Returns (statements: list[str], leftover_body: str)."""
        if not main_name:
            return [], body
        fns = self._extract_top_level_fns(body)
        main_fn = next((f for f in fns if f['name'] == main_name), None)
        if main_fn is None:
            return [], body
        other_fns = [f for f in fns if f['name'] != main_name]
        inlined = self._inline_calls_in_body(main_fn['body_text'], other_fns, uid)
        leftover = body[:main_fn['start']] + body[main_fn['end']:]
        return self._split_rust_statements(inlined), leftover

    # ------------------------------------------------------------------
    # Statement splitting & cross-seed interleave
    # ------------------------------------------------------------------

    def _split_rust_statements(self, code: str):
        """Split a function body into statement units. Compound
        blocks used in statement position (if/for/while/loop/match/unsafe/
        labeled blocks) self-terminate at their closing brace; anything
        that looks like a `let`/`const`/`static` binding keeps accumulating
        until its trailing `;`, since its value may itself be a block
        expression (`let x = match ... { ... };`)."""
        statements = []
        current = []
        brace_depth = paren_depth = bracket_depth = 0
        is_real = self._rust_real_mask(code)
        n = len(code)
        i = 0
        while i < n:
            ch = code[i]
            current.append(ch)
            if not is_real[i]:
                i += 1
                continue
            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            elif ch == '[':
                bracket_depth += 1
            elif ch == ']':
                bracket_depth -= 1
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth <= 0 and paren_depth <= 0 and bracket_depth <= 0:
                    brace_depth = 0
                    stmt_so_far = ''.join(current)
                    is_binding = bool(re.match(r'^\s*(?:let\s|const\s|static\s)', stmt_so_far))
                    if not is_binding:
                        stripped = stmt_so_far.strip()
                        if stripped:
                            statements.append(stripped)
                        current = []
                    i += 1
                    continue
            elif ch == ';' and brace_depth <= 0 and paren_depth <= 0 and bracket_depth <= 0:
                stripped = ''.join(current).strip()
                if stripped:
                    statements.append(stripped)
                current = []
                i += 1
                continue
            i += 1
        leftover = ''.join(current).strip()
        if leftover:
            statements.append(leftover)
        return statements

    def _interleave_preserving_order(self, stmts_a, stmts_b):
        """Riffle-merge two statement streams, keeping each seed's own
        relative order intact (Rust's move/borrow rules make reordering
        within a seed risky; interleaving across seeds is the safe axis)."""
        tagged = []
        ia = ib = 0
        while ia < len(stmts_a) or ib < len(stmts_b):
            if ia >= len(stmts_a):
                tagged.append(('b', stmts_b[ib])); ib += 1
            elif ib >= len(stmts_b):
                tagged.append(('a', stmts_a[ia])); ia += 1
            elif random.random() < 0.5:
                tagged.append(('a', stmts_a[ia])); ia += 1
            else:
                tagged.append(('b', stmts_b[ib])); ib += 1
        return tagged

    def _bridge_across_seeds(self, tagged_stmts):
        """Pick one later `let` in one seed and rebind it from an earlier,
        type-compatible `let` in the other seed — the actual cross-seed
        variable fusion. Only bridges typed `let`s, and only from a
        definition that appears earlier in the merged order."""
        defs = []
        for idx, (origin, stmt) in enumerate(tagged_stmts):
            m = self._LET_TYPE_RE.search(stmt)
            if m:
                norm_t = re.sub(r"&(?:'[a-z_]+\s+)?(?:mut\s+)?", "", m.group(2)).strip()
                defs.append((idx, origin, m.group(1), norm_t))
        if len(defs) < 2:
            return tagged_stmts

        order = list(range(len(defs)))
        random.shuffle(order)
        for k in order:
            tgt_idx, tgt_origin, tgt_name, tgt_type = defs[k]
            earlier_other = [d for d in defs if d[0] < tgt_idx and d[1] != tgt_origin]
            compatible = [d for d in earlier_other if self._types_compatible(d[3], tgt_type)]

            if not self.type_match:
                # Unchanged pre-flag behavior: only ever bridge type-compatible lets.
                candidates = compatible
            elif random.random() < 0.90:
                # Prefer type-compatible, but fall back to any earlier cross-seed
                # let (deliberately type-mismatched) when none is compatible.
                candidates = compatible or earlier_other
            else:
                candidates = earlier_other

            if not candidates:
                continue
            src_idx, src_origin, src_name, src_type = random.choice(candidates)
            needs_clone = self._base_type(src_type) not in self._COPY_TYPES
            rhs = f"{src_name}.clone()" if needs_clone else src_name
            bridge_stmt = f"let {self.bridge_var_name} = {rhs};"
            tagged_stmts.insert(src_idx + 1, (src_origin, bridge_stmt))
            # src_idx < tgt_idx, so inserting before tgt shifts its position by 1.
            new_tgt_idx = tgt_idx + 1

            # Search strictly *after* tgt's own definition line, so we rebind a
            # later use of tgt_name — never the `let tgt_name = ...;` line itself
            # (that would silently discard the binding and dangle later uses).
            for j in range(new_tgt_idx + 1, len(tagged_stmts)):
                o, s = tagged_stmts[j]
                if o == tgt_origin:
                    new_s = self._replace_random_identifier(s, tgt_name, self.bridge_var_name)
                    if new_s is not None:
                        tagged_stmts[j] = (o, new_s)
                        break
            break
        return tagged_stmts

    # ------------------------------------------------------------------

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        # Process A and B: separate crate attrs, use lines, and body
        attrs_a, uses_a, body_a, main_a = self._process_seed(code_a, parent_a.id)
        attrs_b, uses_b, body_b, main_b = self._process_seed(code_b, parent_b.id)

        all_attrs = list(dict.fromkeys(attrs_a + attrs_b))
        all_uses = sorted(set(uses_a + uses_b))

        # ── Inline eligible free-function calls inside each seed's main,
        #    then split into statements ──────────────────────────────────
        main_stmts_a, leftover_a = self._flatten_and_extract_main(body_a, main_a, parent_a.id)
        main_stmts_b, leftover_b = self._flatten_and_extract_main(body_b, main_b, parent_b.id)

        # ── Interleave the two statement streams into one shared scope,
        #    then attempt one type-checked cross-seed variable bridge ─────
        tagged = self._interleave_preserving_order(main_stmts_a, main_stmts_b)
        tagged = self._bridge_across_seeds(tagged)
        merged_main_body = "\n".join(stmt for _, stmt in tagged)

        new_main = (
            "fn main() {\n"
            f"{merged_main_body}\n"
            '    println!("FFL Fusion Done");\n'
            "}"
        )

        # Assemble: #![...] attrs → use lines → leftover items → new main
        parts = []
        if all_attrs:
            parts.append("\n".join(all_attrs))
        if all_uses:
            parts.append("\n".join(all_uses))
        if leftover_a.strip():
            parts.append(f"// Seed A (non-inlined items)\n{leftover_a}")
        if leftover_b.strip():
            parts.append(f"// Seed B (non-inlined items)\n{leftover_b}")
        parts.append(new_main)
        final_content = "\n\n".join(parts)

        return Seed(
            content=final_content,
            metadata={
                "parents":     [parent_a.id, parent_b.id],
                "type":        "rust",
                "description": f"Fused {parent_a.id} + {parent_b.id}",
                "fusion_mode": "stmt_inline",
            }
        )


class RustStructFusionStrategy(RustLexMixin, FusionStrategy):
    """
    "Struct fusion": item-level (struct/enum/trait/impl/fn) fusion for Rust.

    Governing idea: since we never execute the compiled binary, only the
    compiler's own static analyses (name resolution, trait solving,
    coherence checking, monomorphization, layout/discriminant computation)
    are worth stressing — runtime dataflow is irrelevant. Rust's item
    grammar is also recursively self-similar: the crate root, `mod { }`
    bodies, and `fn { }` bodies (including method bodies inside impl/
    trait) are all valid places to declare arbitrary items. So instead of
    a menu of special-case splices, most of what's interesting reduces to
    ONE primitive: take item(s) from one seed and nest them inside a
    container found in the other.

    `impl`/`trait` bodies themselves are NOT general containers (they only
    allow associated items: fn/const/type), so two more operations are
    handled separately because they aren't reducible to nesting:
      - supertrait injection — Rust's nearest analogue to inheritance
        (`trait Foo` -> `trait Foo: OtherTrait`)
      - impl grafting — make a struct/enum from one seed implement a
        trait from the other, auto-stubbing whatever the trait requires
      - generic bound injection — add an extra trait bound from one seed
        onto a generic parameter defined in the other

    This strategy does NOT do statement-level/dataflow fusion at all — it
    only rearranges item definitions.
    """

    def __init__(self, project_root="projects/rust"):
        self.project_root = project_root
        self.mut = RustMutator()

    # ------------------------------------------------------------------
    # Splitting a crate body into top-level items
    # ------------------------------------------------------------------

    _ITEM_KW_RE = re.compile(
        r'(?:pub(?:\([^)]*\))?\s+)?'
        r'(?:(?:async|unsafe|extern(?:\s+"[^"]*")?)\s+){0,2}'
        r'(fn|struct|enum|trait|impl|mod|union|type|static|const)\b'
    )

    _NUMERIC_TYPES = frozenset({
        "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize", "f32", "f64",
    })

    def _extend_for_attrs(self, body: str, start: int) -> int:
        """Walk backward from `start` over blank lines, `#[...]`
        attributes, and doc comments, so an item's span includes its own
        attributes/derives instead of orphaning them."""
        lines = body[:start].splitlines(keepends=True)
        i = len(lines)
        while i > 0:
            s = lines[i - 1].strip()
            if s == '' or s.startswith('#') or s.startswith('//'):
                i -= 1
                continue
            break
        return sum(len(l) for l in lines[:i])

    def _split_top_level_items(self, body: str):
        """Split `body` into an ordered list of dicts covering the ENTIRE
        text (so reconstruction via ''.join(it['text'] for it in items) is
        lossless). Recognized items get kind/name/header/inner; anything
        else (use lines, macro_rules!, stray text) becomes kind='other'
        and is never selected as a nest/graft/bound-inject target."""
        is_real = self._rust_real_mask(body)
        n = len(body)
        depth_before = [0] * n
        depth = 0
        for i in range(n):
            if is_real[i]:
                depth_before[i] = depth
                if body[i] == '{':
                    depth += 1
                elif body[i] == '}':
                    depth = max(0, depth - 1)

        raw_starts = sorted(
            (m.start(), m.group(1)) for m in self._ITEM_KW_RE.finditer(body)
            if is_real[m.start()] and depth_before[m.start()] == 0
        )

        items = []
        cursor = 0
        for start, kind in raw_starts:
            if start < cursor:
                continue
            ext_start = self._extend_for_attrs(body, start)
            name_m = re.match(re.escape(kind) + r'\s*!?\s+([A-Za-z_][A-Za-z0-9_]*)', body[start:start + 200])
            name = name_m.group(1) if name_m else None

            j = start
            found_brace = found_semi = None
            while j < n:
                if is_real[j]:
                    if body[j] == '{':
                        found_brace = j
                        break
                    if body[j] == ';':
                        found_semi = j
                        break
                j += 1
            if found_brace is not None:
                close = self._rust_matching_close(body, is_real, found_brace, '{', '}')
                if close is None:
                    break
                end = close + 1
                header = body[ext_start:found_brace].strip()
                inner = body[found_brace + 1:close]
            elif found_semi is not None:
                end = found_semi + 1
                header = body[ext_start:found_semi].strip()
                inner = None
            else:
                break

            if ext_start > cursor:
                items.append({'kind': 'other', 'text': body[cursor:ext_start]})
            items.append({
                'kind': kind, 'name': name, 'header': header, 'inner': inner,
                'text': body[ext_start:end],
            })
            cursor = end
        if cursor < n:
            items.append({'kind': 'other', 'text': body[cursor:]})
        return items

    def _rebuild_container(self, item: dict, extra_texts: list):
        """Return new item text with `extra_texts` prepended inside its body."""
        new_inner = "\n".join(extra_texts) + "\n" + (item['inner'] or '')
        item = dict(item, inner=new_inner)
        item['text'] = f"{item['header']} {{\n{new_inner}\n}}"
        return item

    # ------------------------------------------------------------------
    # Cross-seed name collision handling
    # ------------------------------------------------------------------

    def _rename_collisions(self, items_a, items_b, uid_b):
        names_a = {it['name'] for it in items_a if it['kind'] != 'other' and it['name']}
        collisions = {it['name'] for it in items_b
                      if it['kind'] != 'other' and it['name'] and it['name'] in names_a}
        if not collisions:
            return items_b
        out = []
        for it in items_b:
            if it['kind'] == 'other':
                out.append(it)
                continue
            text = it['text']
            for name in sorted(collisions, key=len, reverse=True):
                new_name = f"{name}_b{uid_b}"
                text = re.sub(r'(?<![.\w:])' + re.escape(name) + r'(?!\w)', new_name, text)
            new_name_field = f"{it['name']}_b{uid_b}" if it['name'] in collisions else it['name']
            out.append(dict(it, text=text, name=new_name_field))
        return out

    # ------------------------------------------------------------------
    # Operation 1: nesting — the core primitive
    # ------------------------------------------------------------------

    def _op_nest(self, items_a, items_b):
        """Pick a container (fn/mod item with a body) from either seed's
        item list, and move 1-2 whole items from either seed's top-level
        list into it. Moving (not copying) means we never end up with the
        same definition both nested AND still visible at crate root."""
        pools = [('a', items_a), ('b', items_b)]
        containers = [(tag, idx, it) for tag, lst in pools for idx, it in enumerate(lst)
                      if it['kind'] in ('fn', 'mod') and it['inner'] is not None]
        movable = [(tag, idx, it) for tag, lst in pools for idx, it in enumerate(lst)
                   if it['kind'] in ('fn', 'struct', 'enum', 'trait', 'impl', 'mod', 'union')]
        if not containers or len(movable) < 2:
            return items_a, items_b
        c_tag, c_idx, container = random.choice(containers)
        candidates = [m for m in movable if not (m[0] == c_tag and m[1] == c_idx)]
        if not candidates:
            return items_a, items_b
        chosen = random.sample(candidates, min(random.randint(1, 2), len(candidates)))

        # Remove chosen units from their origin lists (mark for deletion by identity).
        remove_a = {idx for tag, idx, _ in chosen if tag == 'a'}
        remove_b = {idx for tag, idx, _ in chosen if tag == 'b'}
        new_a = [it for i, it in enumerate(items_a) if i not in remove_a]
        new_b = [it for i, it in enumerate(items_b) if i not in remove_b]

        # Re-locate the container in its (possibly shrunk) list by identity.
        target_list = new_a if c_tag == 'a' else new_b
        for i, it in enumerate(target_list):
            if it is container:
                target_list[i] = self._rebuild_container(it, [u['text'] for _, _, u in chosen])
                break
        return new_a, new_b

    # ------------------------------------------------------------------
    # Operation 2: supertrait injection (the "inheritance" analogue)
    # ------------------------------------------------------------------

    def _op_supertrait(self, items_a, items_b):
        traits_a = [it for it in items_a if it['kind'] == 'trait']
        traits_b = [it for it in items_b if it['kind'] == 'trait']
        pool = [('a', items_a, it, traits_b) for it in traits_a if traits_b] + \
               [('b', items_b, it, traits_a) for it in traits_b if traits_a]
        if not pool:
            return items_a, items_b
        tag, lst, target, other_traits = random.choice(pool)
        super_name = random.choice(other_traits)['name']
        if not super_name:
            return items_a, items_b
        header = target['header']
        header_wo_generics = re.sub(r'<[^<>]*>', '', header)
        if 'where' in header_wo_generics:
            pre, _, post = header.partition('where')
            has_bound = ':' in re.sub(r'<[^<>]*>', '', pre)
            sep = ' + ' if has_bound else ': '
            new_header = f"{pre.rstrip()}{sep}{super_name} where{post}"
        else:
            has_bound = ':' in header_wo_generics
            sep = ' + ' if has_bound else ': '
            new_header = f"{header}{sep}{super_name}"
        new_target = dict(target, header=new_header, text=f"{new_header} {{\n{target['inner']}\n}}")
        new_lst = [new_target if it is target else it for it in lst]
        return (new_lst, items_b) if tag == 'a' else (items_a, new_lst)

    # ------------------------------------------------------------------
    # Operation 3: impl grafting
    # ------------------------------------------------------------------

    _REQ_FN_RE = re.compile(
        r'\bfn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^;{]+?))?\s*;'
    )
    _REQ_TYPE_RE = re.compile(r'\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=;]+)?\s*;')
    _REQ_CONST_RE = re.compile(r'\bconst\s+([A-Za-z_][A-Za-z0-9_]*)\s*:\s*([^=;]+);')

    def _filler_for_type(self, ty: str) -> str:
        ty = ty.strip()
        base = re.sub(r"^&(?:'[a-z_]+\s+)?(?:mut\s+)?", "", ty)
        if base in self._NUMERIC_TYPES:
            return "0"
        if base == "bool":
            return "false"
        if base in ("str",) or ty in ("&str", "&'static str"):
            return '""'
        return "Default::default()"

    def _stub_impl_body(self, trait_inner: str) -> str:
        members = []
        for m in self._REQ_FN_RE.finditer(trait_inner):
            name, generics, params, ret = m.group(1), m.group(2) or '', m.group(3), m.group(4)
            ret_ty = (ret or '').strip()
            body = "todo!()" if not ret_ty or ret_ty == '()' else "todo!()"
            members.append(f"fn {name}{generics}({params}) -> {ret_ty or '()'} {{ {body} }}"
                            if ret_ty else f"fn {name}{generics}({params}) {{ {body} }}")
        for m in self._REQ_TYPE_RE.finditer(trait_inner):
            members.append(f"type {m.group(1)} = ();")
        for m in self._REQ_CONST_RE.finditer(trait_inner):
            ty = m.group(2).strip()
            members.append(f"const {m.group(1)}: {ty} = {self._filler_for_type(ty)};")
        return "\n".join(members)

    def _op_impl_graft(self, items_a, items_b):
        types_a = [it for it in items_a if it['kind'] in ('struct', 'enum')]
        types_b = [it for it in items_b if it['kind'] in ('struct', 'enum')]
        traits_a = [it for it in items_a if it['kind'] == 'trait' and '<' not in it['header']]
        traits_b = [it for it in items_b if it['kind'] == 'trait' and '<' not in it['header']]
        pool = [('a', items_a, t, tr) for t in types_a for tr in traits_b] + \
               [('b', items_b, t, tr) for t in types_b for tr in traits_a]
        if not pool:
            return items_a, items_b
        tag, lst, target_type, trait_item = random.choice(pool)
        if not target_type['name'] or not trait_item['name']:
            return items_a, items_b
        stub_body = self._stub_impl_body(trait_item['inner'] or '')
        impl_text = f"impl {trait_item['name']} for {target_type['name']} {{\n{stub_body}\n}}"
        new_impl = {'kind': 'impl', 'name': None, 'header': impl_text.split('{', 1)[0].strip(),
                    'inner': stub_body, 'text': impl_text}
        new_lst = lst + [new_impl]
        return (new_lst, items_b) if tag == 'a' else (items_a, new_lst)

    # ------------------------------------------------------------------
    # Operation 4: generic bound injection
    # ------------------------------------------------------------------

    def _op_bound_inject(self, items_a, items_b):
        traits_a = [it for it in items_a if it['kind'] == 'trait']
        traits_b = [it for it in items_b if it['kind'] == 'trait']
        candidates_a = [it for it in items_a
                        if it['kind'] in ('fn', 'struct', 'trait', 'impl') and re.search(r'<[A-Z]\w*', it['header'])]
        candidates_b = [it for it in items_b
                        if it['kind'] in ('fn', 'struct', 'trait', 'impl') and re.search(r'<[A-Z]\w*', it['header'])]
        pool = [('a', items_a, it, traits_b) for it in candidates_a if traits_b] + \
               [('b', items_b, it, traits_a) for it in candidates_b if traits_a]
        if not pool:
            return items_a, items_b
        tag, lst, target, other_traits = random.choice(pool)
        trait_name = random.choice(other_traits)['name']
        if not trait_name:
            return items_a, items_b
        m = re.search(r'<([A-Z]\w*)(\s*:\s*[^,<>]+)?([,>])', target['header'])
        if not m:
            return items_a, items_b
        if m.group(2):
            new_header = target['header'][:m.end(2)] + f" + {trait_name}" + target['header'][m.end(2):]
        else:
            new_header = target['header'][:m.end(1)] + f": {trait_name}" + target['header'][m.end(1):]
        new_target = dict(target, header=new_header)
        if target['inner'] is not None:
            new_target['text'] = f"{new_header} {{\n{target['inner']}\n}}"
        else:
            # semicolon item — reconstruct from the original text's tail
            tail = target['text'][len(target['header']):]
            new_target['text'] = new_header + tail
        new_lst = [new_target if it is target else it for it in lst]
        return (new_lst, items_b) if tag == 'a' else (items_a, new_lst)

    # ------------------------------------------------------------------

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content
        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        uid_a = re.sub(r'[^a-zA-Z0-9_]', '_', parent_a.id)
        uid_b = re.sub(r'[^a-zA-Z0-9_]', '_', parent_b.id)

        # Separate crate-level attrs / use-imports from the rest, same as
        # RustFusionStrategy — these must be hoisted above everything else.
        crate_attrs, use_lines, body_lines_a, body_lines_b = [], [], [], []
        for code, body_lines in ((code_a, body_lines_a), (code_b, body_lines_b)):
            for line in code.splitlines():
                stripped = line.strip()
                if stripped.startswith("#!"):
                    crate_attrs.append(line)
                elif stripped.startswith("use ") or stripped.startswith("extern crate "):
                    use_lines.append(line)
                else:
                    body_lines.append(line)
        all_attrs = list(dict.fromkeys(crate_attrs))
        all_uses = sorted(set(use_lines))

        items_a = self._split_top_level_items("\n".join(body_lines_a))
        items_b = self._split_top_level_items("\n".join(body_lines_b))
        items_b = self._rename_collisions(items_a, items_b, uid_b)

        # Apply a random combination of the 4 operations — at least one,
        # since that's the whole point of this strategy.
        ops = [self._op_nest, self._op_supertrait, self._op_impl_graft, self._op_bound_inject]
        applied_any = False
        for op in ops:
            if random.random() < 0.5:
                items_a, items_b = op(items_a, items_b)
                applied_any = True
        if not applied_any:
            items_a, items_b = self._op_nest(items_a, items_b)

        body = "\n".join(it['text'] for it in items_a) + "\n" + "\n".join(it['text'] for it in items_b)

        if not re.search(r'\bfn\s+main\s*\(', body):
            body += "\nfn main() {}\n"

        parts = []
        if all_attrs:
            parts.append("\n".join(all_attrs))
        if all_uses:
            parts.append("\n".join(all_uses))
        parts.append(body)
        final_content = "\n\n".join(parts)

        return Seed(
            content=final_content,
            metadata={
                "parents":     [parent_a.id, parent_b.id],
                "type":        "rust",
                "description": f"Struct-fused {parent_a.id} + {parent_b.id}",
                "fusion_mode": "struct",
            }
        )


class HaskellFusionStrategy(FusionStrategy):
    """
    Haskell-Specific Fusion Strategy.

    `self.dataflow_fusion` is the only mode `fuse()`/`fuse_bidirectional()`
    pick (dataflow fusion, below). A second, concurrent-racing mode
    ('state_ab'/'state_ba') is also implemented in `_build_fused_test()`
    but has no caller anymore — kept for reference/reuse, not currently
    wired to any CLI flag or entry point. See HaskellStateFusionStrategy
    below for the actual `--state-fusion` strategy, which is a different
    mechanism entirely: it grafts a donor's continuation into a profiled
    most-complex-state point rather than racing two whole programs
    concurrently.

    - Dataflow fusion ('df_ab'/'df_ba'): a value harvested from the source
      seed is written into a shared top-level CAF (an `unsafePerformIO`'d
      IORef) and spliced into a numeric-literal position in the
      destination seed via `fromInteger`, so the splice type-unifies with
      almost any Num context. Exercises GHC's constant folding, CAF
      sharing, and unsafePerformIO duplication-under-optimization paths.

    - Concurrent-racing mode ('state_ab'/'state_ba', currently unreachable):
      builds a harness where both seeds' entry actions would run
      *concurrently* via forkIO, racing on one shared top-level
      IORef/MVar/TVar cell (kind picked from whichever stateful
      construct, if any, the seeds themselves already use).
      NOTE: projects/haskell/driver.py is compile-only (`ghc -fno-code`,
      never links or runs the program), so today this harness is only
      ever type-checked — it exercises GHC's handling of concurrent-
      looking code (closures captured by forkIO, CAF/unsafePerformIO
      typing, MVar/STM API usage), not actual runtime races. The fused
      program is still shaped for real races (forkIO + shared cell + a
      System.Timeout-bounded join) in case the driver ever switches back
      to executing.

    Both modes: strip module headers, merge imports/pragmas, rename `main`
    to a unique per-seed entry point (synthesizing one if a seed had none),
    rename colliding top-level declarations in B, and wrap each entry
    action in `try` so one seed's ordinary runtime exception doesn't
    prevent the other from running (relevant if the driver ever executes
    fused output; harmless dead code under compile-only fuzzing).
    """

    _HS_KEYWORDS = frozenset({
        'data', 'newtype', 'type', 'class', 'instance', 'where', 'let', 'in',
        'do', 'if', 'then', 'else', 'case', 'of', 'import', 'module',
        'deriving', 'infixl', 'infixr', 'infix', 'foreign', 'default',
        'family', 'forall', 'main', 'True', 'False', 'Nothing', 'Just',
        'Left', 'Right',
    })

    # Typeclass methods / Prelude staples — never rename these even if a
    # seed happens to define a same-named top-level binding, since renaming
    # inside an `instance ... where` block would break the instance.
    _HS_SKIP_RENAME = frozenset({
        'show', 'showsPrec', 'showList', 'read', 'readsPrec',
        'compare', 'max', 'min', 'fmap', 'pure', 'return',
        'mempty', 'mappend', 'toEnum', 'fromEnum', 'succ', 'pred',
        'minBound', 'maxBound', 'foldr', 'foldl', 'foldMap', 'traverse',
        'sequence', 'id', 'const', 'map', 'filter', 'length', 'head',
        'tail', 'init', 'last', 'div', 'mod', 'quot', 'rem', 'negate',
        'abs', 'signum', 'fromInteger', 'toInteger', 'fromRational',
        'toRational',
    })

    _MODULE_HEADER_RE = re.compile(r'^\s*module\s+.*?\bwhere\b\s*', re.DOTALL)
    _IMPORT_LINE_RE = re.compile(r'^\s*import\s+.*$', re.MULTILINE)
    _PRAGMA_RE = re.compile(r'\{-#.*?#-\}', re.DOTALL)
    _MAIN_DEF_RE = re.compile(r'^main\s*(?:::|=)', re.MULTILINE)
    _TOPLEVEL_RE = re.compile(
        r"^([a-z_][A-Za-z0-9_']*)\s*(?:::|[^=\n]*=)"
        r"|^(?:data|newtype|type)\s+([A-Z][A-Za-z0-9_']*)"
        r"|^class\s+(?:.*=>\s*)?([A-Z][A-Za-z0-9_']*)",
        re.MULTILINE,
    )
    _INT_LIT_RE = re.compile(r"(?<![A-Za-z0-9_.'])[0-9]+(?![A-Za-z0-9_.'])")

    # Per-kind templates for the shared mutable cell used by state fusion
    # (and, fixed to 'ioref', by dataflow fusion). Kept generic over
    # IORef/MVar/TVar so `state_handles` metadata can steer which
    # primitive gets exercised.
    _STATE_KIND_INFO = {
        'ioref': {
            'type': 'IORef Integer',
            'create': 'unsafePerformIO (newIORef 0)',
            'write': lambda shared, v: f"writeIORef {shared} ({v})",
            'read': lambda shared: f"readIORef {shared}",
            'import': "import Data.IORef",
        },
        'mvar': {
            'type': 'MVar Integer',
            'create': 'unsafePerformIO (newMVar 0)',
            'write': lambda shared, v: f"modifyMVar_ {shared} (\\_ -> return ({v}))",
            'read': lambda shared: f"readMVar {shared}",
            'import': "import Control.Concurrent.MVar",
        },
        'tvar': {
            'type': 'TVar Integer',
            'create': 'unsafePerformIO (newTVarIO 0)',
            'write': lambda shared, v: f"atomically (writeTVar {shared} ({v}))",
            'read': lambda shared: f"readTVarIO {shared}",
            'import': "import Control.Concurrent.STM",
        },
    }

    _HS_IDENT_RE = re.compile(r"\b[a-z_][A-Za-z0-9_']*\b")

    def __init__(self, project_root="projects/haskell", type_match: bool = False,
                 lightweight: bool = False):
        self.project_root = project_root
        self.mut = HaskellMutator()
        self.dataflow_fusion = True
        self.state_fusion = False
        self.all_fusion = False
        # --dataflow-type-match: 90% of the time, splice into an integer-
        # literal site as before (always type-safe via `fromInteger`),
        # falling back to a random identifier when body has none; 10% of
        # the time, skip straight to substituting a random identifier's
        # occurrences with the raw bridge expression — deliberately risking
        # a type mismatch to stress GHC's type-checker diagnostics. Off by
        # default — behavior is unchanged from before this flag existed
        # (always the type-safe integer-literal splice, or no-op).
        self.type_match = type_match
        # Dataflow fusion here is already on-the-fly (_harvest_int_literal
        # scans the seed directly, no cached metadata), so this is a no-op
        # for fuse() — it only exists so HaskellStateFusionStrategy (which
        # inherits this __init__) can read self.lightweight.
        self.lightweight = lightweight

    # ------------------------------------------------------------------
    # Seed processing
    # ------------------------------------------------------------------

    def _process_seed(self, code: str, uid: str) -> dict:
        pragmas = [p.strip() for p in self._PRAGMA_RE.findall(code)]
        code = self._PRAGMA_RE.sub('', code)
        code = self._MODULE_HEADER_RE.sub('', code, count=1)

        imports = [ln.strip() for ln in self._IMPORT_LINE_RE.findall(code)]
        code = self._IMPORT_LINE_RE.sub('', code)

        had_main = bool(self._MAIN_DEF_RE.search(code))
        main_name = f"fflMain_{uid}"
        code = re.sub(r'\bmain\b', main_name, code)

        toplevel = set()
        for m in self._TOPLEVEL_RE.finditer(code):
            name = m.group(1) or m.group(2) or m.group(3)
            if name:
                toplevel.add(name)
        toplevel -= self._HS_KEYWORDS
        toplevel.discard(main_name)

        return {
            "pragmas": pragmas,
            "imports": imports,
            "body": code.strip(),
            "had_main": had_main,
            "main_name": main_name,
            "toplevel_names": toplevel,
        }

    def _rename_collisions(self, names_a: set, body_b: str, names_b: set, uid_b: str) -> str:
        collisions = (names_b & names_a) - self._HS_SKIP_RENAME - self._HS_KEYWORDS
        if not collisions:
            return body_b
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid_b)
        for name in sorted(collisions, key=len, reverse=True):
            body_b = re.sub(r'\b' + re.escape(name) + r'\b', f"{name}_b{uid_safe}", body_b)
        return body_b

    def _build_entry_action(self, name: str, had_main: bool, nullary_bindings: list) -> str:
        """
        Return an extra top-level decl (possibly empty) providing
        `name :: IO ()`. If the seed already declared `main`, it was
        already renamed to `name` by _process_seed and nothing more is
        needed. Otherwise synthesize a small IO action that forces one of
        the seed's pure top-level bindings (if any) through `show`, so GHC
        still exercises the seed's declarations even without an original
        `main`.
        """
        if had_main:
            return ""
        if nullary_bindings:
            probe = random.choice(nullary_bindings)
            return (
                f"{name} :: IO ()\n"
                f"{name} = do\n"
                f"  r <- try (evaluate (length (show ({probe})))) :: IO (Either SomeException Int)\n"
                f"  case r of\n"
                f"    Left e -> putStrLn (\"probe error: \" ++ show (e :: SomeException))\n"
                f"    Right n -> putStrLn (\"probe len: \" ++ show n)\n"
            )
        return f"{name} :: IO ()\n{name} = return ()\n"

    # ------------------------------------------------------------------
    # Bridge helpers
    # ------------------------------------------------------------------

    def _harvest_int_literal(self, body: str, default: str = "1") -> str:
        matches = self._INT_LIT_RE.findall(body)
        return random.choice(matches) if matches else default

    def _splice_numeric_bridge(self, body: str, bridge_read_expr: str) -> str:
        """Replace a weighted-random number of integer-literal occurrences
        (see pick_occurrence_count) in body with a read from the shared
        cell, wrapped in `fromInteger` so it type-unifies with (almost) any
        Num context the literal originally sat in.

        --dataflow-type-match (self.type_match): 90% of the time, splice
        into integer-literal sites as above, falling back to a random
        identifier occurrence when body has none; 10% of the time, replace
        occurrences of a random non-keyword identifier in body with the raw
        bridge expression instead — deliberately risking a type mismatch
        (the point of the flag: stress GHC's type-checker diagnostics)
        rather than always picking the type-safe numeric slot. With the
        flag off, behavior is unchanged: always the integer-literal splice,
        or a no-op if body has none.
        """
        use_numeric = not (self.type_match and random.random() >= 0.90)

        if use_numeric:
            matches = list(self._INT_LIT_RE.finditer(body))
            if matches:
                n = min(pick_occurrence_count(), len(matches))
                chosen = sorted(random.sample(matches, n), key=lambda m: m.start(), reverse=True)
                for m in chosen:
                    body = body[:m.start()] + f"(fromInteger ({bridge_read_expr}))" + body[m.end():]
                return body
            if not self.type_match:
                return body
            # No integer literal to splice into — fall through to the
            # identifier-substitution path below instead of giving up.

        idents = [m.group(0) for m in self._HS_IDENT_RE.finditer(body)
                  if m.group(0) not in self._HS_KEYWORDS]
        if not idents:
            return body
        target = random.choice(idents)
        return replace_word_occurrences(body, target, f"({bridge_read_expr})")

    def _pick_shared_kind(self, meta_a: dict, meta_b: dict) -> str:
        for meta in (meta_a, meta_b):
            handles = meta.get('state_handles') or []
            if handles:
                return handles[0].get('kind', 'ioref')
        return random.choice(['ioref', 'mvar', 'tvar'])

    # ------------------------------------------------------------------
    # Fusion build
    # ------------------------------------------------------------------

    def _build_fused_test(self, parent_a: Seed, parent_b: Seed, mode: str) -> Seed:
        code_a, code_b = parent_a.content, parent_b.content
        meta_a, meta_b = parent_a.metadata, parent_b.metadata

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        uid_a = re.sub(r'[^a-zA-Z0-9]', '_', str(parent_a.id))
        uid_b = re.sub(r'[^a-zA-Z0-9]', '_', str(parent_b.id))

        proc_a = self._process_seed(code_a, uid_a)
        proc_b = self._process_seed(code_b, uid_b)

        body_a = proc_a['body']
        body_b = self._rename_collisions(proc_a['toplevel_names'], proc_b['body'], proc_b['toplevel_names'], uid_b)

        entry_a = self._build_entry_action(proc_a['main_name'], proc_a['had_main'], meta_a.get('nullary_bindings', []))
        entry_b = self._build_entry_action(proc_b['main_name'], proc_b['had_main'], meta_b.get('nullary_bindings', []))

        pragmas = sorted(set(proc_a['pragmas']) | set(proc_b['pragmas']))
        seed_imports = sorted(set(proc_a['imports']) | set(proc_b['imports']))
        base_imports = ["import Control.Exception (SomeException, evaluate, try)"]

        is_state = mode.startswith('state_')
        src_first = mode.endswith('_ab')  # True: A is the bridge/state "source"

        src_body, dst_body = (body_a, body_b) if src_first else (body_b, body_a)
        src_name = proc_a['main_name'] if src_first else proc_b['main_name']
        dst_name = proc_b['main_name'] if src_first else proc_a['main_name']
        meta_src = meta_a if src_first else meta_b

        if is_state:
            base_imports += [
                "import Control.Concurrent (forkIO)",
                "import Control.Concurrent.MVar (newEmptyMVar, putMVar, takeMVar)",
                "import System.Timeout (timeout)",
                "import System.IO.Unsafe (unsafePerformIO)",
            ]
            kind = self._pick_shared_kind(meta_a, meta_b)
            info = self._STATE_KIND_INFO[kind]
            base_imports.append(info['import'])

            shared = f"fflShared_{uid_a}_{uid_b}"
            shared_decl = (
                f"{{-# NOINLINE {shared} #-}}\n"
                f"{shared} :: {info['type']}\n"
                f"{shared} = {info['create']}\n"
            )
            seed_val = self._harvest_int_literal(src_body)
            dst_lit = self._harvest_int_literal(dst_body, seed_val)
            write_src = info['write'](shared, seed_val)
            read_expr = info['read'](shared)

            main_body = (
                "main :: IO ()\n"
                "main = do\n"
                f"  {write_src}\n"
                "  doneA <- newEmptyMVar\n"
                "  doneB <- newEmptyMVar\n"
                "  _ <- forkIO (do\n"
                f"    _ <- (try ({src_name}) :: IO (Either SomeException ()))\n"
                "    putMVar doneA ())\n"
                "  _ <- forkIO (do\n"
                f"    v <- {read_expr}\n"
                f"    {info['write'](shared, f'v + ({dst_lit})')}\n"
                f"    _ <- (try ({dst_name}) :: IO (Either SomeException ()))\n"
                "    putMVar doneB ())\n"
                "  _ <- timeout (3 * 1000000) (takeMVar doneA)\n"
                "  _ <- timeout (3 * 1000000) (takeMVar doneB)\n"
                f"  finalVal <- {read_expr}\n"
                "  putStrLn (\"FFL shared state final: \" ++ show finalVal)\n"
            )
        else:
            base_imports += ["import Data.IORef", "import System.IO.Unsafe (unsafePerformIO)"]
            info = self._STATE_KIND_INFO['ioref']
            shared = f"fflBridge_{uid_a}_{uid_b}"
            shared_decl = (
                f"{{-# NOINLINE {shared} #-}}\n"
                f"{shared} :: IORef Integer\n"
                f"{shared} = {info['create']}\n"
            )

            nullary_src = meta_src.get('nullary_bindings', [])
            if nullary_src and random.random() < 0.5:
                write_val = f"toInteger (length (show ({random.choice(nullary_src)})))"
            else:
                write_val = self._harvest_int_literal(src_body)
            write_src = info['write'](shared, write_val)
            read_expr = info['read'](shared)

            # The splice site is a *pure* expression position in dst_body,
            # but readIORef is an IO action — force it out-of-line via
            # unsafePerformIO so the substituted text still type-checks.
            dst_body = self._splice_numeric_bridge(dst_body, f"unsafePerformIO ({read_expr})")
            if src_first:
                body_a, body_b = src_body, dst_body
            else:
                body_a, body_b = dst_body, src_body

            main_body = (
                "main :: IO ()\n"
                "main = do\n"
                f"  {write_src}\n"
                f"  r1 <- (try ({src_name}) :: IO (Either SomeException ()))\n"
                "  case r1 of\n"
                "    Left e -> putStrLn (\"A: \" ++ show (e :: SomeException))\n"
                "    Right _ -> return ()\n"
                f"  r2 <- (try ({dst_name}) :: IO (Either SomeException ()))\n"
                "  case r2 of\n"
                "    Left e -> putStrLn (\"B: \" ++ show (e :: SomeException))\n"
                "    Right _ -> return ()\n"
                f"  finalVal <- {read_expr}\n"
                "  putStrLn (\"FFL bridge final: \" ++ show finalVal)\n"
            )

        all_imports = sorted(set(base_imports) | set(seed_imports))

        sections = ["module Main where"]
        if pragmas:
            # LANGUAGE pragmas must appear before the module header.
            sections.insert(0, "\n".join(pragmas))
        sections.append("\n".join(all_imports))
        sections.append(shared_decl)
        sections.append(f"-- === Seed A: {parent_a.id} ===\n{body_a}")
        if entry_a:
            sections.append(entry_a)
        sections.append(f"-- === Seed B: {parent_b.id} ===\n{body_b}")
        if entry_b:
            sections.append(entry_b)
        sections.append(main_body)

        final_content = "\n\n".join(s for s in sections if s.strip())

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "haskell",
                "mode": mode,
                "description": f"Fused {parent_a.id} + {parent_b.id} ({mode})",
            },
        )

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        if self.state_fusion:
            mode = random.choice(['state_ab', 'state_ba'])
        else:
            mode = random.choice(['df_ab', 'df_ba'])
        return self._build_fused_test(parent_a, parent_b, mode)

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """Produce both A->B and B->A variants for the active fusion kind."""
        if self.state_fusion:
            return [
                self._build_fused_test(parent_a, parent_b, 'state_ab'),
                self._build_fused_test(parent_a, parent_b, 'state_ba'),
            ]
        return [
            self._build_fused_test(parent_a, parent_b, 'df_ab'),
            self._build_fused_test(parent_a, parent_b, 'df_ba'),
        ]


class HaskellStateFusionStrategy(HaskellFusionStrategy):
    """
    State fusion for Haskell using core/state_analysis.py's profiled
    splice points (`hClose`, `fromIntegral`, a `catch`/`throw`
    boundary), distinct from HaskellFusionStrategy's own 'state_ab'/
    'state_ba' modes (a *concurrent* forkIO race on a shared cell — see
    that class's docstring). This strategy instead grafts the donor's
    continuation directly into the host's body as text, at the host's
    splice point, then runs the fused single action —
    structurally closer to how the other languages' state fusion works.

    Splice-point selection: unlike the other languages (a live-
    variable-count analysis, see core/state_analysis.py), Haskell always
    just picks any random line — pattern-bound names (function arguments,
    case alternatives, do-binds) are Haskell's real equivalent of "many
    variables," but its layout-based scoping (where/let-in/do blocks
    scoped by indentation, not braces) makes tracking where those go out
    of scope too unreliable to be worth it, so this doesn't try; it's
    independent of --pre-analysis (no richer analysis to fall back to
    here regardless of the flag).

    CAVEAT: this is the least-verified of the state-fusion strategies —
    no local `ghc` was available to check the grafted output against
    Haskell's indentation-sensitive layout rule (unlike Clang, which was
    checked against real g++). It's built on the same reindent-to-host-
    column mechanism that's structurally sound for Python's layout rule,
    but Haskell's layout algorithm has more edge cases (e.g. `where`
    clauses, multi-equation function definitions) that a donor
    continuation could still trip. Also, projects/haskell/driver.py is
    compile-only (`ghc -fno-code`), so even a successful run here only
    exercises the type-checker, not actual execution.
    """

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint

        host_code, donor_code = host.content, donor.content
        if self.mut:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)

        uid_host = re.sub(r'[^a-zA-Z0-9]', '_', str(host.id))
        uid_donor = re.sub(r'[^a-zA-Z0-9]', '_', str(donor.id))

        proc_host = self._process_seed(host_code, uid_host)
        proc_donor = self._process_seed(donor_code, uid_donor)

        body_host = proc_host['body']
        body_donor = self._rename_collisions(proc_host['toplevel_names'], proc_donor['body'],
                                              proc_donor['toplevel_names'], uid_donor)

        entry_host = self._build_entry_action(proc_host['main_name'], proc_host['had_main'],
                                               host.metadata.get('nullary_bindings', []))

        # Always random-line selection (see class docstring) — independent
        # of self.lightweight/--pre-analysis, not just when it's off.
        host_point = pick_state_point(body_host, "haskell", lightweight=True)
        donor_point = pick_state_point(body_donor, "haskell", lightweight=True)
        if host_point is None:
            lines = body_host.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        fused_body = graft_continuation(body_host, body_donor, host_point, donor_point)

        pragmas = sorted(set(proc_host['pragmas']) | set(proc_donor['pragmas']))
        seed_imports = sorted(set(proc_host['imports']) | set(proc_donor['imports']))
        base_imports = ["import Control.Exception (SomeException, evaluate, try)"]
        all_imports = sorted(set(base_imports) | set(seed_imports))

        main_body = (
            "main :: IO ()\n"
            "main = do\n"
            f"  r <- (try ({proc_host['main_name']}) :: IO (Either SomeException ()))\n"
            "  case r of\n"
            "    Left e -> putStrLn (\"error: \" ++ show (e :: SomeException))\n"
            "    Right _ -> return ()\n"
        )

        sections = ["module Main where"]
        if pragmas:
            sections.insert(0, "\n".join(pragmas))
        sections.append("\n".join(all_imports))
        sections.append(f"-- === State-fused: host {host.id} <- donor {donor.id} ({host_point.category}) ===\n{fused_body}")
        if entry_host:
            sections.append(entry_host)
        sections.append(main_body)

        final_content = "\n\n".join(s for s in sections if s.strip())

        return Seed(content=final_content, metadata={
            "parents": [host.id, donor.id],
            "type": "haskell",
            "mode": f"state_{host_point.category}_{direction}",
            "description": f"State-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


class HaskellDeclarationFusionStrategy(HaskellFusionStrategy):
    """
    Declaration fusion for Haskell: injects a donor-declared typeclass as
    an extra superclass constraint on a host typeclass
    (`class Foo a where` -> `class DonorClass a => Foo a where`) — the
    Haskell analogue of Rust's supertrait injection. GHC's instance
    resolution and superclass-satisfaction checking is a purely static,
    declare-time process (no call needed): a class with an unsatisfiable
    or self-referential superclass context fails to even be accepted,
    independent of whether anything ever uses it.
    """

    _HS_CLASS_HEADER_RE = re.compile(
        r'^(?P<indent>[ \t]*)class\s+'
        r'(?P<ctx>\([^)]*\)\s*=>\s*)?'
        r'(?P<name>[A-Za-z_]\w*)\s+(?P<tyvar>[a-z]\w*)\b'
        r'[^\n]*?\bwhere\b',
        re.MULTILINE,
    )

    def _inject_superclass(self, code: str, donor_class_name: str):
        matches = list(self._HS_CLASS_HEADER_RE.finditer(code))
        if not matches:
            return code, False
        m = random.choice(matches)
        tyvar = m.group('tyvar')
        if m.group('ctx'):
            ctx = m.group('ctx')
            close_paren = ctx.rfind(')')
            if close_paren == -1:
                return code, False
            new_ctx = ctx[:close_paren] + f", {donor_class_name} {tyvar}" + ctx[close_paren:]
            new_code = code[:m.start('ctx')] + new_ctx + code[m.end('ctx'):]
        else:
            insert_pos = m.start('name')
            new_code = code[:insert_pos] + f"{donor_class_name} {tyvar} => " + code[insert_pos:]
        return new_code, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_code, donor_code = host.content, donor.content
        if self.mut:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)

        uid_host = re.sub(r'[^a-zA-Z0-9]', '_', str(host.id))
        uid_donor = re.sub(r'[^a-zA-Z0-9]', '_', str(donor.id))

        proc_host = self._process_seed(host_code, uid_host)
        proc_donor = self._process_seed(donor_code, uid_donor)

        body_donor = self._rename_collisions(proc_host['toplevel_names'], proc_donor['body'],
                                              proc_donor['toplevel_names'], uid_donor)
        body_host = proc_host['body']

        donor_class_names = [m.group('name') for m in self._HS_CLASS_HEADER_RE.finditer(body_donor)]
        fused_host, applied = body_host, False
        if donor_class_names:
            donor_name = random.choice(donor_class_names)
            fused_host, applied = self._inject_superclass(body_host, donor_name)

        entry_host = self._build_entry_action(proc_host['main_name'], proc_host['had_main'],
                                               host.metadata.get('nullary_bindings', []))

        pragmas = sorted(set(proc_host['pragmas']) | set(proc_donor['pragmas']))
        seed_imports = sorted(set(proc_host['imports']) | set(proc_donor['imports']))
        base_imports = ["import Control.Exception (SomeException, evaluate, try)"]
        all_imports = sorted(set(base_imports) | set(seed_imports))

        main_body = (
            "main :: IO ()\n"
            "main = do\n"
            f"  r <- (try ({proc_host['main_name']}) :: IO (Either SomeException ()))\n"
            "  case r of\n"
            "    Left e -> putStrLn (\"error: \" ++ show (e :: SomeException))\n"
            "    Right _ -> return ()\n"
        )

        sections = ["module Main where"]
        if pragmas:
            sections.insert(0, "\n".join(pragmas))
        sections.append("\n".join(all_imports))
        sections.append(f"-- === Donor typeclasses: {donor.id} ===\n{body_donor}")
        sections.append(f"-- === Declaration-fused host: {host.id} ===\n{fused_host}")
        if entry_host:
            sections.append(entry_host)
        sections.append(main_body)

        final_content = "\n\n".join(s for s in sections if s.strip())

        return Seed(content=final_content, metadata={
            "parents": [host.id, donor.id],
            "type": "haskell",
            "mode": f"decl_{'superclass' if applied else 'none'}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


# ==========================================
# Clang / C-C++ Specific Fusion Strategy
# ==========================================

class ClangFusionStrategy(GenericDataflowStrategy):
    """
    Clang (C/C++) fusion strategy. Supports two modes, mirroring PHP:

    - Dataflow fusion (default): concatenates the two seeds and links them
      with a bridge variable (`interleave_code_blocks`, inherited from
      GenericDataflowStrategy) — a global `static long ffl_fusion = (long)(x);`
      whose value is threaded into an occurrence of a variable from the other
      seed. Cross-seed variable references are intentionally best-effort
      (may not be a compile-time constant) — clang emits a diagnostic and
      keeps going rather than crashing, which is fine; we're stressing the
      frontend/diagnostics paths, not compiling working programs.

    - Statement fusion: splits both seeds into top-level "statement" units
      (a unit is a full function/struct/enum/class definition, a single
      declaration, or a preprocessor directive — never partially split), then
      interleaves them via dependency-graph topological sort (variable/
      function/type def-use), with optional injection of a top-level unit
      into a compound block body.
    """

    # Best-effort C/C++ variable-declaration type extractor — a heuristic,
    # not a real parser, used only to bias --dataflow-type-match's same-type
    # preference. Matches `<base-type> [*...] name (=|;|,|\))`.
    _C_VAR_DECL_RE = re.compile(
        r'\b(_Bool|bool|char|short(?:\s+int)?|int|long(?:\s+long)?(?:\s+int)?|'
        r'float|double|long\s+double|'
        r'unsigned(?:\s+(?:char|short|int|long(?:\s+long)?))?|'
        r'signed(?:\s+(?:char|short|int|long(?:\s+long)?))?|'
        r'size_t|ssize_t|int8_t|int16_t|int32_t|int64_t|'
        r'uint8_t|uint16_t|uint32_t|uint64_t)'
        r'\s*(\*{0,2})\s*([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*(?:=(?!=)|;|,|\))'
    )

    def __init__(self, project_root="projects/clang", type_match: bool = False,
                 lightweight: bool = False):
        super().__init__(mutator=BaseMutator(), lightweight=lightweight)
        self.project_root = project_root
        self.bridge_var_name = "ffl_fusion"
        self.mutation = True
        self.stmt_fusion = False
        self.dataflow_fusion = True
        self.all_fusion = False
        # --dataflow-type-match: bias bridge-variable selection toward a
        # same-declared-type pair 90% of the time (falling back to random
        # when no match exists), and pick purely at random the other 10%.
        # Off by default — behavior is unchanged from before this flag existed.
        self.type_match = type_match

    def format_assignment(self, lhs: str, rhs: str) -> str:
        # Top-level declaration — valid C/C++ syntax even when `rhs` turns out
        # not to be a real compile-time constant (clang just diagnoses it).
        return f"static long {lhs} = (long)({rhs});"

    def _lightweight_vars(self, code: str) -> List[str]:
        """Reuse the --dataflow-type-match variable-declaration extractor
        on the fly instead of parse-time dataflow1/dataflow2 metadata."""
        return list(self._infer_c_var_types(code).keys())

    def _lightweight_replace(self, code: str, var: str, bridge: str) -> str:
        return self._replace_random_occurrence_word(code, var, bridge)

    @staticmethod
    def _replace_random_occurrence_word(s: str, old: str, new: str) -> str:
        """Word-boundary-safe replacement. Unlike PHP's `$`-sigil or MLIR's
        `%`-sigil variables, bare C identifiers have no unambiguous prefix,
        so a naive substring replace (replace_random_occurrence) can corrupt
        unrelated identifiers or even keywords that merely contain `old` as
        a substring (e.g. replacing bare `r` would turn `struct` into
        `styuct`, or `for` into `foy`). Delegates to the shared
        replace_word_occurrences (also used by Swift) for the actual
        weighted-occurrence-count replacement."""
        return replace_word_occurrences(s, old, new)

    def _infer_c_var_types(self, code: str) -> Dict[str, str]:
        """Best-effort name -> declared-type map (e.g. 'int', 'char*') from
        simple declarations in `code`. Heuristic, not a real parser — good
        enough to bias --dataflow-type-match's same-type preference."""
        types = {}
        for m in self._C_VAR_DECL_RE.finditer(code):
            base_type, stars, name = m.group(1), m.group(2), m.group(3)
            types[name] = re.sub(r'\s+', ' ', base_type.strip()) + stars
        return types

    def _pick_var2(self, var1, code1, code2, dataflow2, fallback):
        """Choose the bridge-substitution target in B. Under
        --dataflow-type-match: 90% of the time, search *all* of B's
        dataflow-tracked variables for one whose inferred declared type
        matches var1's, falling back to `fallback()` (the pre-existing
        random pick) when var1's type is unknown or no match exists; the
        other 10% always calls `fallback()`. With the flag off, always
        calls `fallback()` — identical to pre-flag behavior."""
        if self.type_match and random.random() < 0.90:
            t1 = self._infer_c_var_types(code1).get(var1)
            if t1:
                all_b_vars = {v for group in dataflow2 for v in group}
                types2 = self._infer_c_var_types(code2)
                same_type = [v for v in all_b_vars if types2.get(v) == t1]
                if same_type:
                    return random.choice(same_type)
        return fallback()

    def interleave_code_blocks(self, code1, code2, dataflow1, dataflow2, extra_flows=None):
        """Override of GenericDataflowStrategy.interleave_code_blocks that
        substitutes via word boundaries instead of raw substring replace —
        see _replace_random_occurrence_word. Bridge-target selection goes
        through _pick_var2 so --dataflow-type-match can bias it."""
        if self.lightweight:
            return self._lightweight_bridge(code1, code2)
        if not dataflow1 or not dataflow2:
            return code1, code2

        if extra_flows:
            dataflow1 = dataflow1 + extra_flows

        bridge = self.bridge_var_name

        if random.choice([True, False]):
            try:
                group1 = random.choice(dataflow1)
                var1 = random.choice(group1)
                group2 = random.choice(dataflow2)
                var2 = self._pick_var2(var1, code1, code2, dataflow2, lambda: random.choice(group2))
                bridge_stmt = self.format_assignment(bridge, var1)
                code1 += f"\n{bridge_stmt}\n"
                code2 = self._replace_random_occurrence_word(code2, var2, bridge)
            except IndexError:
                pass
            return code1, code2

        max_df1 = max(dataflow1, key=len) if dataflow1 else []
        max_df2 = max(dataflow2, key=len) if dataflow2 else []

        if max_df1 and max_df2:
            var1 = random.choice(max_df1)
            var2 = self._pick_var2(var1, code1, code2, dataflow2, lambda: random.choice(max_df2))
            bridge_stmt = self.format_assignment(bridge, var1)
            code1 += f"\n{bridge_stmt}\n"
            code2 = self._replace_random_occurrence_word(code2, var2, bridge)

        return code1, code2

    # ── Statement Fusion helpers ──────────────────────────────────

    _C_KEYWORDS = frozenset({
        "auto", "break", "case", "char", "const", "continue", "default", "do",
        "double", "else", "enum", "extern", "float", "for", "goto", "if",
        "inline", "int", "long", "register", "restrict", "return", "short",
        "signed", "sizeof", "static", "struct", "switch", "typedef", "union",
        "unsigned", "void", "volatile", "while",
        "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic",
        "_Imaginary", "_Noreturn", "_Static_assert", "_Thread_local",
        "alignas", "alignof", "and", "and_eq", "asm", "bitand", "bitor",
        "bool", "catch", "class", "compl", "concept", "const_cast",
        "consteval", "constexpr", "constinit", "co_await", "co_return",
        "co_yield", "decltype", "delete", "dynamic_cast", "explicit",
        "export", "false", "friend", "mutable", "namespace", "new",
        "noexcept", "not", "not_eq", "nullptr", "operator", "or", "or_eq",
        "private", "protected", "public", "reinterpret_cast", "requires",
        "static_assert", "static_cast", "template", "this", "thread_local",
        "throw", "true", "try", "typeid", "typename", "using", "virtual",
        "wchar_t", "xor", "xor_eq", "NULL",
    })
    _C_CONTROL_KW = frozenset({"if", "for", "while", "switch", "catch"})

    _C_VAR_ASSIGN_RE = re.compile(r'\b([A-Za-z_]\w*)\s*=(?!=)')
    _C_FUNC_DEF_RE = re.compile(r'\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*(?:const\s*)?\{')
    _C_TYPE_DEF_RE = re.compile(r'\b(?:struct|class|union|enum)\s+([A-Za-z_]\w*)')
    _C_IDENT_RE = re.compile(r'\b([A-Za-z_]\w*)\b')
    _C_NORM_STR = re.compile(r'"(?:[^"\\]|\\.)*"')
    _C_NORM_CHAR = re.compile(r"'(?:[^'\\]|\\.)*'")
    _C_NORM_NUM = re.compile(r'\b0[xX][0-9a-fA-F]+[uUlL]*\b|\b\d+(?:\.\d+)?[uUlLfF]*\b')

    _C_BLOCK_HEAD_RE = re.compile(
        r'^\s*(?:'
        r'if\s*\(|else\b|for\s*\(|while\s*\(|do\b|switch\s*\('
        r'|try\b|catch\s*\('
        r'|struct\s+\w+|class\s+\w+|union\s+\w+|enum\s+\w+|namespace\s+\w+'
        r'|[A-Za-z_][\w:\*&<>,\s]*?\s+[A-Za-z_]\w*\s*\([^;{}]*\)\s*(?:const\s*)?'
        r')'
    )

    _CONTINUATION_RE = re.compile(r'^\s*(?:else\b|while\s*\(|catch\s*\()')

    @classmethod
    def _split_statements(cls, code: str) -> List[str]:
        """Split C/C++ code into top-level statement units, respecting
        brace/paren depth, string/char literals, comments, and preprocessor
        directives (kept as a single atomic line, honouring `\\`-continuation).
        Compound blocks (functions, structs, control structures, including
        do/while and if/else chains) are kept as single units."""
        statements: List[str] = []
        current: List[str] = []
        brace_depth = 0
        paren_depth = 0
        in_sq = False
        in_dq = False
        escaped = False
        i = 0
        n = len(code)

        def _is_blank(buf: List[str]) -> bool:
            return not ''.join(buf).strip()

        def _lookahead_is_continuation(pos: int) -> bool:
            rest = code[pos:]
            return bool(cls._CONTINUATION_RE.match(rest))

        while i < n:
            ch = code[i]
            nch = code[i + 1] if i + 1 < n else ''

            if escaped:
                current.append(ch)
                escaped = False
                i += 1
                continue
            if ch == '\\' and (in_sq or in_dq):
                current.append(ch)
                escaped = True
                i += 1
                continue
            if in_sq:
                current.append(ch)
                if ch == "'":
                    in_sq = False
                i += 1
                continue
            if in_dq:
                current.append(ch)
                if ch == '"':
                    in_dq = False
                i += 1
                continue

            # ── preprocessor directive: consume the whole (possibly
            #    continued) line as one atomic statement ──
            if (ch == '#' and brace_depth == 0 and paren_depth == 0
                    and _is_blank(current)):
                current = []
                while i < n:
                    c = code[i]
                    current.append(c)
                    if c == '\\' and i + 1 < n and code[i + 1] == '\n':
                        current.append(code[i + 1])
                        i += 2
                        continue
                    if c == '\n':
                        i += 1
                        break
                    i += 1
                stmt = ''.join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
                continue

            # ── comments ──
            if ch == '/' and nch == '/':
                while i < n and code[i] != '\n':
                    current.append(code[i])
                    i += 1
                continue
            if ch == '/' and nch == '*':
                current.append(ch)
                i += 1
                while i < n:
                    current.append(code[i])
                    if code[i] == '*' and i + 1 < n and code[i + 1] == '/':
                        current.append('/')
                        i += 2
                        break
                    i += 1
                continue

            if ch == "'":
                in_sq = True
                current.append(ch)
                i += 1
                continue
            if ch == '"':
                in_dq = True
                current.append(ch)
                i += 1
                continue

            current.append(ch)

            if ch == '(':
                paren_depth += 1
            elif ch == ')':
                paren_depth -= 1
            elif ch == '{':
                brace_depth += 1
            elif ch == '}':
                brace_depth -= 1
                if brace_depth <= 0 and paren_depth <= 0:
                    brace_depth = 0
                    j = i + 1
                    while j < n and code[j] in (' ', '\t', '\n', '\r'):
                        j += 1
                    if _lookahead_is_continuation(j):
                        i += 1
                        continue
                    stmt = ''.join(current).strip()
                    if stmt:
                        statements.append(stmt)
                    current = []
                    i += 1
                    continue
            elif ch == ';' and brace_depth <= 0 and paren_depth <= 0:
                stmt = ''.join(current).strip()
                if stmt:
                    statements.append(stmt)
                current = []
                i += 1
                continue

            i += 1

        leftover = ''.join(current).strip()
        if leftover:
            statements.append(leftover)
        return statements

    def _stmt_defines_uses(self, stmt: str):
        """Return (defines, func_or_type_defines, uses) for a statement unit.
        `func_or_type_defines` (functions/structs/classes/enums/unions) is a
        subset of `defines` eligible for cross-seed dependency edges — plain
        variable assignments stay seed-local (cross-seed variable refs are
        intentionally invalid/best-effort, same rationale as PHP fusion)."""
        func_defines = set()
        for m in self._C_FUNC_DEF_RE.finditer(stmt):
            name = m.group(1)
            if name not in self._C_KEYWORDS and name not in self._C_CONTROL_KW:
                func_defines.add(name)
        for m in self._C_TYPE_DEF_RE.finditer(stmt):
            func_defines.add(m.group(1))

        var_defines = set()
        for m in self._C_VAR_ASSIGN_RE.finditer(stmt):
            name = m.group(1)
            if name not in self._C_KEYWORDS:
                var_defines.add(name)

        uses = {t for t in self._C_IDENT_RE.findall(stmt) if t not in self._C_KEYWORDS}

        return (var_defines | func_defines), func_defines, uses

    def _normalize_stmt(self, stmt: str) -> str:
        """Normalize for similarity comparison: blank out string/char
        literals, numeric literals, and non-keyword identifiers."""
        s = self._C_NORM_STR.sub('"_"', stmt)
        s = self._C_NORM_CHAR.sub("'_'", s)
        s = self._C_NORM_NUM.sub('0', s)
        s = self._C_IDENT_RE.sub(
            lambda m: m.group(0) if m.group(0) in self._C_KEYWORDS else '_', s)
        return s

    @staticmethod
    def _token_similarity(norm_a: str, norm_b: str) -> float:
        toks_a = set(norm_a.split())
        toks_b = set(norm_b.split())
        if not toks_a and not toks_b:
            return 1.0
        union = toks_a | toks_b
        return len(toks_a & toks_b) / len(union) if union else 0.0

    def _dependency_graph_interleave(self, stmts_a: List[str], stmts_b: List[str]) -> List[str]:
        """Interleave statement units from seed A and seed B using a
        dependency-graph topological sort with token-similarity tie-breaking
        (same approach as PHP statement fusion)."""
        tagged = [(s, 'a') for s in stmts_a] + [(s, 'b') for s in stmts_b]
        n = len(tagged)
        if n == 0:
            return []

        info = []
        for stmt, origin in tagged:
            defines, func_defines, uses = self._stmt_defines_uses(stmt)
            info.append({
                'stmt': stmt, 'origin': origin, 'defines': defines,
                'func_defines': func_defines, 'uses': uses,
                'norm': self._normalize_stmt(stmt),
            })

        def_map: Dict[str, List[Tuple[int, str]]] = {}
        for i, inf in enumerate(info):
            for name in inf['defines']:
                def_map.setdefault(name, []).append((i, inf['origin']))

        deps = [set() for _ in range(n)]
        for j, inf_j in enumerate(info):
            for name in inf_j['uses']:
                if name not in def_map:
                    continue
                for di, d_origin in def_map[name]:
                    if di == j:
                        continue
                    if d_origin == inf_j['origin']:
                        deps[j].add(di)
                    elif name in info[di]['func_defines']:
                        # Cross-seed edge only for function/type names — a
                        # call to a function defined in the other seed must
                        # come after its definition.
                        deps[j].add(di)

        emitted = [False] * n
        result: List[str] = []
        in_degree = [len(d) for d in deps]
        dependents = [[] for _ in range(n)]
        for j in range(n):
            for di in deps[j]:
                dependents[di].append(j)

        ready = [i for i in range(n) if in_degree[i] == 0]
        last_norm = ""
        while ready:
            if not result:
                pick_idx = random.choice(ready)
            else:
                scored = sorted(
                    ((self._token_similarity(last_norm, info[ri]['norm']), ri) for ri in ready),
                    key=lambda x: -x[0])
                top_k = min(3, len(scored))
                pick_idx = random.choice([s[1] for s in scored[:top_k]])

            ready.remove(pick_idx)
            emitted[pick_idx] = True
            result.append(info[pick_idx]['stmt'])
            last_norm = info[pick_idx]['norm']

            for dep_j in dependents[pick_idx]:
                in_degree[dep_j] -= 1
                if in_degree[dep_j] == 0 and not emitted[dep_j]:
                    ready.append(dep_j)

        for i in range(n):
            if not emitted[i]:
                result.append(info[i]['stmt'])

        return result

    def _stmt_cross_replace_variable(self, code: str, vars_a: List[str], vars_b: List[str]) -> str:
        if not vars_a or not vars_b:
            return code
        var_b = random.choice(vars_b)
        var_a = random.choice(vars_a)
        if var_a == var_b:
            return code
        return self._replace_random_occurrence_word(code, var_b, var_a)

    @staticmethod
    def _find_outermost_brace_body(stmt: str):
        in_sq = in_dq = escaped = False
        depth = 0
        body_start = -1
        for i, ch in enumerate(stmt):
            if escaped:
                escaped = False
                continue
            if ch == '\\' and (in_sq or in_dq):
                escaped = True
                continue
            if in_sq:
                if ch == "'":
                    in_sq = False
                continue
            if in_dq:
                if ch == '"':
                    in_dq = False
                continue
            if ch == "'":
                in_sq = True
                continue
            if ch == '"':
                in_dq = True
                continue
            if ch == '{':
                depth += 1
                if depth == 1:
                    body_start = i + 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and body_start != -1:
                    return (body_start, i)
        return None

    def _inject_into_block(self, stmts: List[str]) -> List[str]:
        candidates = []
        atomic = []
        for idx, s in enumerate(stmts):
            if self._C_BLOCK_HEAD_RE.match(s) and '{' in s:
                span = self._find_outermost_brace_body(s)
                if span:
                    body_start, body_end = span
                    body = s[body_start:body_end].strip()
                    if len(body) > 5:
                        candidates.append((idx, body_start, body_end))
            elif s.strip():
                atomic.append((idx, s))

        if not candidates or not atomic:
            return stmts

        target_idx, body_start, body_end = random.choice(candidates)
        target = stmts[target_idx]
        body = target[body_start:body_end]

        donors = [(i, s) for i, s in atomic if i != target_idx]
        if not donors:
            return stmts
        n_inject = min(random.randint(1, 3), len(donors))
        chosen = random.sample(donors, n_inject)
        inject_stmts = [s for _, s in chosen]

        body_lines = body.split('\n')
        insert_pos = random.randint(0, len(body_lines))

        indent = "    "
        for ln in body_lines:
            stripped = ln.lstrip()
            if stripped:
                indent = ln[:len(ln) - len(stripped)]
                break

        injected = [indent + s.strip() for s in inject_stmts]
        new_body_lines = body_lines[:insert_pos] + injected + body_lines[insert_pos:]
        new_body = '\n'.join(new_body_lines)

        new_stmt = target[:body_start] + new_body + target[body_end:]
        result = list(stmts)
        result[target_idx] = new_stmt
        return result

    def _statement_fuse(self, clean1: str, clean2: str,
                         vars1: List[str], vars2: List[str]) -> str:
        stmts_a = self._split_statements(clean1)
        stmts_b = self._split_statements(clean2)

        if not stmts_a and not stmts_b:
            return clean1 + '\n' + clean2

        interleaved = self._dependency_graph_interleave(stmts_a, stmts_b)

        if random.random() < 0.3:
            interleaved = self._inject_into_block(interleaved)

        fused_code = '\n'.join(interleaved)
        fused_code = self._stmt_cross_replace_variable(fused_code, vars1, vars2)
        return fused_code

    # ── Top-level name-conflict resolution ────────────────────────

    def _extract_top_level_names(self, code: str):
        names = set(self._C_FUNC_DEF_RE.findall(code)) | set(self._C_TYPE_DEF_RE.findall(code))
        return names - self._C_KEYWORDS - self._C_CONTROL_KW

    def _resolve_name_conflicts(self, code_a: str, code_b: str) -> str:
        """Rename top-level functions/types in code_b that clash with names
        defined in code_a, to reduce duplicate-symbol diagnostics."""
        conflicts = self._extract_top_level_names(code_a) & self._extract_top_level_names(code_b)
        if not conflicts:
            return code_b
        result = code_b
        for name in sorted(conflicts, key=len, reverse=True):
            result = re.sub(
                rf'(?<![\w.:>])\b{re.escape(name)}\b',
                name + '_ffl',
                result,
            )
        return result

    # ── Entry points ───────────────────────────────────────────────

    def _build_fused_test(self, parent_a: Seed, parent_b: Seed, mode: str) -> Seed:
        code1 = parent_a.content
        code2 = parent_b.content
        meta1 = parent_a.metadata
        meta2 = parent_b.metadata
        variable1 = meta1.get('variables', [])
        variable2 = meta2.get('variables', [])
        dataflow1 = meta1.get('dataflows', [])
        dataflow2 = meta2.get('dataflows', [])

        if self.mutation:
            code1 = self.mut.mutate(code1)
            code2 = self.mut.mutate(code2)

        code2 = self._resolve_name_conflicts(code1, code2)

        if mode == 'stmt_ab':
            fused = self._statement_fuse(code1, code2, variable1, variable2)
        elif mode == 'stmt_ba':
            fused = self._statement_fuse(code2, code1, variable2, variable1)
        elif mode == 'df_ab':
            new_code1, new_code2 = self.interleave_code_blocks(code1, code2, dataflow1, dataflow2)
            fused = f"{new_code1}\n{new_code2}"
        elif mode == 'df_ba':
            new_code2, new_code1 = self.interleave_code_blocks(code2, code1, dataflow2, dataflow1)
            fused = f"{new_code1}\n{new_code2}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Prefer C++/Obj-C metadata from either parent so the driver picks
        # clang++ when either side actually needs it.
        cxx_exts = ('.cpp', '.cc', '.cxx', '.mm')
        ext = meta1.get('extension', '.c')
        seed_type = meta1.get('type', 'c')
        if meta2.get('extension') in cxx_exts:
            ext = meta2.get('extension')
            seed_type = meta2.get('type', 'cpp')

        return Seed(content=fused, metadata={
            "parents": [parent_a.id, parent_b.id],
            "type": seed_type,
            "extension": ext,
            "mode": mode,
            "description": f"Fused {parent_a.id} + {parent_b.id} ({mode})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        mode = 'stmt_ab' if self.stmt_fusion else 'df_ab'
        return self._build_fused_test(parent_a, parent_b, mode)

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """Produce both A->B and B->A variants for the active fusion kind."""
        if self.stmt_fusion:
            return [
                self._build_fused_test(parent_a, parent_b, 'stmt_ab'),
                self._build_fused_test(parent_a, parent_b, 'stmt_ba'),
            ]
        return [
            self._build_fused_test(parent_a, parent_b, 'df_ab'),
            self._build_fused_test(parent_a, parent_b, 'df_ba'),
        ]


# ==========================================
# Clang Declaration Fusion Strategy
# ==========================================

class ClangDeclarationFusionStrategy(ClangFusionStrategy):
    """
    Declaration fusion for C/C++: confuses *declarations* rather than
    *dataflow*. Unlike ClangFusionStrategy's dataflow/statement modes,
    which need a runtime value from one seed to actually reach and trigger
    a bug in the other, these bugs are triggered simply by declaring or
    instantiating the fused definition — no execution required, which
    makes this the more effective lever on clang's frontend/sema (name
    resolution, overload/template instantiation, class layout) where
    dataflow fusion offers comparatively little.

    Three primitives, mirroring the paper's "extensible declaration
    expressions" for C++ (base class lists, template parameters):

    - base_class:    make a class/struct in the host seed additionally
                      inherit from a class/struct declared in the donor
                      seed (`class Foo : public Bar { ... }` ->
                      `class Foo : public Bar, public <donor> { ... }`).
    - template_param: add a defaulted template parameter to a template
                      class/struct in the host, defaulting to a type
                      declared in the donor seed.
    - item_nest:      nest a whole struct/class/enum declaration from the
                      donor seed inside a namespace/class/function body
                      found in the host (same core primitive as Rust's
                      RustStructFusionStrategy, adapted to C++ syntax).

    Reuses ClangFusionStrategy's literal-aware statement splitter
    (_split_statements) and name-conflict resolution rather than
    reimplementing a C++ parser.
    """

    _CLASS_HEADER_RE = re.compile(
        r'(?P<template>template\s*<[^>]*>\s*)?'
        r'\b(?P<kind>class|struct)\s+(?P<name>[A-Za-z_]\w*)'
        r'(?P<bases>\s*:\s*[^{;]+)?'
        r'\s*\{'
    )

    def __init__(self, project_root="projects/clang"):
        super().__init__(project_root=project_root)

    # ── Declaration-site discovery ──────────────────────────────────

    @classmethod
    def _extract_type_names(cls, code: str):
        """name -> True if declared as a template (so any reference to it as
        a base class or default-template-argument needs `<...>` args)."""
        names: Dict[str, bool] = {}
        for m in cls._CLASS_HEADER_RE.finditer(code):
            name = m.group('name')
            if name in cls._C_KEYWORDS or name in cls._C_CONTROL_KW:
                continue
            names[name] = names.get(name, False) or bool(m.group('template'))
        for name in cls._C_TYPE_DEF_RE.findall(code):
            if name in cls._C_KEYWORDS or name in cls._C_CONTROL_KW:
                continue
            names.setdefault(name, False)
        return names

    @staticmethod
    def _donor_type_ref(name: str, is_template: bool) -> str:
        # A template needs args to be used as a base class or a type
        # argument (`Container` alone isn't a type, `Container<int>` is) —
        # otherwise every fusion that happens to pick a template donor
        # would deterministically fail with the same boring diagnostic.
        return f"{name}<int>" if is_template else name

    @classmethod
    def _donor_type_items(cls, code: str):
        """Top-level struct/class/enum declaration units from `code`,
        suitable for nesting whole into another seed's container."""
        items = []
        for stmt in cls._split_statements(code):
            head = stmt.lstrip()[:200]
            if re.match(r'(?:template\s*<[^>]*>\s*)?\b(?:struct|class|enum)\b', head):
                # _split_statements emits the closing '}' of a struct/class/
                # enum as the end of the unit but treats the mandatory
                # trailing ';' as its own separate (bare) statement — so a
                # unit lifted out on its own is missing it. Always ensured
                # here since nesting/nesting-adjacent code takes each item
                # out of that surrounding statement-list context.
                fixed = stmt if stmt.rstrip().endswith(';') else stmt.rstrip() + ';'
                items.append(fixed)
        return items

    @classmethod
    def has_injectable_declaration(cls, code: str) -> bool:
        """True if `code` has at least one struct/class/enum a donor could
        contribute — either as a base-class/template-default reference
        (_extract_type_names, which also picks up bare forward
        declarations) or as a whole nestable unit (_donor_type_items).
        Mirrors exactly what _build_declaration_fused_test checks before
        setting `applied`, so pre-analysis's cached has_declaration flag
        (core/dryrun.py) never drifts from what this strategy actually
        looks for."""
        return bool(cls._extract_type_names(code)) or bool(cls._donor_type_items(code))

    def is_viable_pair(self, parent_a: Seed, parent_b: Seed) -> bool:
        """Declaration fusion only does something on a pair when BOTH seeds
        can donate a declaration — fuse_bidirectional tries each seed as
        donor in turn (ab and ba), so a seed with none is dead weight in
        either role. Without --pre-analysis's has_declaration metadata
        (get_strategies already disables this strategy entirely in that
        case, so this branch shouldn't normally run), falls back to
        scanning both seeds on the spot rather than assuming viability."""
        meta_a, meta_b = parent_a.metadata or {}, parent_b.metadata or {}
        if "has_declaration" in meta_a and "has_declaration" in meta_b:
            return bool(meta_a["has_declaration"]) and bool(meta_b["has_declaration"])
        return (self.has_injectable_declaration(parent_a.content)
                and self.has_injectable_declaration(parent_b.content))

    # ── Primitive 1: base class list injection ──────────────────────

    def _inject_base_class(self, stmt: str, donor_name: str):
        matches = list(self._CLASS_HEADER_RE.finditer(stmt))
        if not matches:
            return stmt, False
        m = random.choice(matches)
        brace_pos = m.end() - 1  # index of the matched '{'
        if m.group('bases'):
            new_stmt = stmt[:brace_pos].rstrip() + f", public {donor_name} " + stmt[brace_pos:]
        else:
            new_stmt = stmt[:brace_pos].rstrip() + f" : public {donor_name} " + stmt[brace_pos:]
        return new_stmt, True

    # ── Primitive 2: template parameter injection ────────────────────

    def _inject_template_param(self, stmt: str, donor_name: str):
        matches = [m for m in self._CLASS_HEADER_RE.finditer(stmt) if m.group('template')]
        if not matches:
            return stmt, False
        m = random.choice(matches)
        tmpl = m.group('template')
        close_rel = tmpl.rfind('>')
        if close_rel == -1:
            return stmt, False
        insert_at = m.start('template') + close_rel
        new_param = f", typename _FflDeclT = {donor_name}"
        new_stmt = stmt[:insert_at] + new_param + stmt[insert_at:]
        return new_stmt, True

    # ── Primitive 3: item nesting ─────────────────────────────────────

    def _nest_declaration(self, host_stmts: List[str], donor_item: str):
        candidates = []
        for idx, s in enumerate(host_stmts):
            if self._C_BLOCK_HEAD_RE.match(s) and '{' in s:
                span = self._find_outermost_brace_body(s)
                if span:
                    candidates.append((idx, span))
        if not candidates:
            return host_stmts, False
        idx, (body_start, body_end) = random.choice(candidates)
        target = host_stmts[idx]
        new_stmt = target[:body_end] + "\n" + donor_item + "\n" + target[body_end:]
        result = list(host_stmts)
        result[idx] = new_stmt
        return result, True

    # ── Entry points ───────────────────────────────────────────────

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_code = host.content
        donor_code = donor.content
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)

        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        donor_types = self._extract_type_names(donor_code)
        host_stmts = self._split_statements(host_code)

        technique = random.choice(['base_class', 'template_param', 'item_nest'])
        applied = False

        if technique in ('base_class', 'template_param') and donor_types:
            donor_name, is_tmpl = random.choice(list(donor_types.items()))
            donor_ref = self._donor_type_ref(donor_name, is_tmpl)
            targets = list(range(len(host_stmts)))
            random.shuffle(targets)
            for idx in targets:
                if technique == 'base_class':
                    new_stmt, ok = self._inject_base_class(host_stmts[idx], donor_ref)
                else:
                    new_stmt, ok = self._inject_template_param(host_stmts[idx], donor_ref)
                if ok:
                    host_stmts[idx] = new_stmt
                    applied = True
                    break

        if not applied:
            donor_items = self._donor_type_items(donor_code)
            if donor_items:
                donor_item = random.choice(donor_items)
                host_stmts, applied = self._nest_declaration(host_stmts, donor_item)
                technique = 'item_nest'

        fused_host = '\n'.join(host_stmts)
        # Donor code goes FIRST: a base-class or default-template-argument
        # reference to a type needs that type already declared earlier in
        # the translation unit, same as any ordinary C++ forward-reference
        # rule. (For item_nest this duplicates the donor's declaration at
        # top level in addition to the nested copy — harmless, just extra
        # surface for redeclaration/ODR diagnostics.)
        fused = f"{donor_code}\n{fused_host}"

        cxx_exts = ('.cpp', '.cc', '.cxx', '.mm')
        meta_a, meta_b = host.metadata, donor.metadata
        ext = meta_a.get('extension', '.cpp')
        seed_type = meta_a.get('type', 'cpp')
        if meta_b.get('extension') in cxx_exts:
            ext = meta_b.get('extension')
            seed_type = meta_b.get('type', 'cpp')
        if ext not in cxx_exts:
            # Base-class/template injection is C++-only syntax
            ext, seed_type = '.cpp', 'cpp'

        return Seed(content=fused, metadata={
            "parents": [host.id, donor.id],
            "type": seed_type,
            "extension": ext,
            "mode": f"decl_{technique}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({technique}, {direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, 'ab')

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """A hosts a declaration referring into B, and vice versa."""
        return [
            self._build_declaration_fused_test(parent_a, parent_b, 'ab'),
            self._build_declaration_fused_test(parent_b, parent_a, 'ba'),
        ]


class ClangStateFusionStrategy(ClangFusionStrategy):
    """
    State fusion for C/C++ (core/state_analysis.py): grafts one seed's
    continuation into the other's state at a profiled most complex state
    (`free()`/destructor call, an explicit cast, a try/catch boundary)
    instead of bridging a single value or interleaving whole statements by
    dependency graph. Complements ClangFusionStrategy's existing modes.
    """

    _INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*(?:include|import)\b.*$', re.MULTILINE)

    @classmethod
    def _extract_includes_and_body(cls, code: str):
        includes = cls._INCLUDE_RE.findall(code)
        body = cls._INCLUDE_RE.sub('', code)
        return body, includes

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint, truncate_to_balanced

        host_code, donor_code = host.content, donor.content
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)
        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        # #include lines must be hoisted out of the graft: they're part of
        # the donor's preamble (before its state point), so a naive graft
        # would silently drop them and the continuation's identifiers would
        # fail to resolve.
        host_body, host_includes = self._extract_includes_and_body(host_code)
        donor_body, donor_includes = self._extract_includes_and_body(donor_code)
        all_includes = sorted(set(i.strip() for i in host_includes) | set(i.strip() for i in donor_includes))

        host_point = pick_state_point(host_body, "clang", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_body, "clang", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_body.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        donor_lines = donor_body.splitlines()
        start_idx = (donor_point.line_idx + 1) if donor_point else 0
        end_idx = truncate_to_balanced(donor_body, start_idx, "clang")
        truncated_donor = "\n".join(donor_lines[:end_idx])

        fused_body = graft_continuation(host_body, truncated_donor, host_point, donor_point)
        fused = "\n".join(all_includes) + "\n" + fused_body

        cxx_exts = ('.cpp', '.cc', '.cxx', '.mm')
        meta_a, meta_b = host.metadata, donor.metadata
        ext = meta_a.get('extension', '.c')
        seed_type = meta_a.get('type', 'c')
        if meta_b.get('extension') in cxx_exts:
            ext = meta_b.get('extension')
            seed_type = meta_b.get('type', 'cpp')

        return Seed(content=fused, metadata={
            "parents": [host.id, donor.id],
            "type": seed_type,
            "extension": ext,
            "mode": f"state_{host_point.category}_{direction}",
            "description": f"State-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


# ==========================================
# Flang / Fortran Specific Fusion Strategy
# ==========================================

class FlangFusionStrategy(GenericDataflowStrategy):
    """
    Flang (Fortran) fusion strategy. Same two modes as ClangFusionStrategy,
    adapted for Fortran's very different structure:

    - No braces — blocks are opened/closed by keyword pairs (IF...THEN /
      END IF, DO / END DO, SUBROUTINE ... / END SUBROUTINE, ...). Depth
      tracking here uses two counters instead of one: `udepth` (nesting of
      PROGRAM/MODULE/SUBROUTINE/FUNCTION/BLOCK DATA "program units") and
      `idepth` (nesting of inner control blocks — IF/DO/SELECT/...).
      Almost all Fortran source lives inside a program unit (unlike C,
      where bare top-level statements are common), so the granularity
      statement fusion actually interleaves is the *body* of the first
      program unit (udepth==1, idepth==0) — a nested IF/DO block or a
      second, CONTAINS-nested unit is kept atomic, exactly like a
      compound-statement block in the C/PHP splitters.

    - No unconditional "append a global declaration at file scope" trick
      for dataflow fusion's bridge variable — a bare statement after the
      last END is not valid Fortran (there is no such thing as top-level
      executable code outside any program unit). Instead the bridge
      assignment is inserted just before the outermost unit's closing END.

    - String literals use doubled-quote escaping ('' / "") instead of
      backslash escaping.
    """

    # Best-effort Fortran variable-declaration type extractor — a
    # heuristic, not a real parser, used only to bias --dataflow-type-
    # match's same-type preference. Matches `<type>[(kind)] :: name[, ...]`.
    _FORTRAN_VAR_DECL_RE = re.compile(
        r'\b(integer|real|double\s+precision|complex|logical|character)\b'
        r'\s*(\([^)]*\)|\*\d+)?\s*(?:,[^:]*)?::\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)',
        re.IGNORECASE,
    )

    def __init__(self, project_root="projects/flang", type_match: bool = False,
                 lightweight: bool = False):
        super().__init__(mutator=BaseMutator(), lightweight=lightweight)
        self.project_root = project_root
        self.bridge_var_name = "ffl_fusion"
        self.mutation = True
        self.stmt_fusion = False
        self.dataflow_fusion = True
        self.all_fusion = False
        # --dataflow-type-match: bias bridge-variable selection toward a
        # same-declared-type pair 90% of the time (falling back to random
        # when no match exists), and pick purely at random the other 10%.
        # Off by default — behavior is unchanged from before this flag existed.
        self.type_match = type_match

    def format_assignment(self, lhs: str, rhs: str) -> str:
        return f"{lhs} = ({rhs})"

    def _lightweight_vars(self, code: str) -> List[str]:
        """On-the-fly declared-name scan (original case, unlike
        _infer_fortran_var_types's uppercased keys) instead of parse-time
        dataflow1/dataflow2 metadata."""
        names = []
        for m in self._FORTRAN_VAR_DECL_RE.finditer(code):
            for name in m.group(3).split(','):
                name = re.split(r'[\s(]', name.strip(), 1)[0]
                if name:
                    names.append(name)
        return names

    def _lightweight_replace(self, code: str, var: str, bridge: str) -> str:
        return self._replace_random_occurrence_word(code, var, bridge)

    def _infer_fortran_var_types(self, code: str) -> Dict[str, str]:
        """Best-effort name -> declared-type map (e.g. 'integer', 'real')
        from simple `TYPE :: name` declarations. Heuristic, not a real
        parser — good enough to bias --dataflow-type-match's same-type
        preference. Keyed case-insensitively (Fortran identifiers are)."""
        types = {}
        for m in self._FORTRAN_VAR_DECL_RE.finditer(code):
            base_type, kind = m.group(1), m.group(2)
            norm = re.sub(r'\s+', ' ', base_type.strip().lower()) + (kind or '')
            for name in m.group(3).split(','):
                types[name.strip().upper()] = norm
        return types

    def _pick_var2(self, var1, code1, code2, dataflow2, fallback):
        """Choose the bridge-substitution target in B. Under
        --dataflow-type-match: 90% of the time, search *all* of B's
        dataflow-tracked variables for one whose inferred declared type
        matches var1's, falling back to `fallback()` (the pre-existing
        random pick) when var1's type is unknown or no match exists; the
        other 10% always calls `fallback()`. With the flag off, always
        calls `fallback()` — identical to pre-flag behavior."""
        if self.type_match and random.random() < 0.90:
            t1 = self._infer_fortran_var_types(code1).get(var1.upper())
            if t1:
                all_b_vars = {v for group in dataflow2 for v in group}
                types2 = self._infer_fortran_var_types(code2)
                same_type = [v for v in all_b_vars if types2.get(v.upper()) == t1]
                if same_type:
                    return random.choice(same_type)
        return fallback()

    # ── Shared lexical helpers ──────────────────────────────────────

    @staticmethod
    def _strip_comment(line: str) -> str:
        """Strip a trailing '! comment', respecting string literals with
        Fortran's doubled-quote escaping ('' inside '...', "" inside "...")."""
        in_sq = in_dq = False
        i, n = 0, len(line)
        while i < n:
            ch = line[i]
            if in_sq:
                if ch == "'":
                    if i + 1 < n and line[i + 1] == "'":
                        i += 2
                        continue
                    in_sq = False
                i += 1
                continue
            if in_dq:
                if ch == '"':
                    if i + 1 < n and line[i + 1] == '"':
                        i += 2
                        continue
                    in_dq = False
                i += 1
                continue
            if ch == "'":
                in_sq = True
                i += 1
                continue
            if ch == '"':
                in_dq = True
                i += 1
                continue
            if ch == '!':
                return line[:i]
            i += 1
        return line

    @classmethod
    def _join_continuations(cls, code: str) -> List[str]:
        """Merge free-form continuation lines (trailing '&', outside strings
        and comments) into logical lines — each returned entry may itself
        contain embedded '\\n' for a multi-physical-line statement, original
        text preserved."""
        physical = code.split('\n')
        logical: List[str] = []
        buf: List[str] = []
        in_sq = in_dq = False
        for line in physical:
            j, ll = 0, len(line)
            code_end = ll
            local_sq, local_dq = in_sq, in_dq
            while j < ll:
                c = line[j]
                if local_sq:
                    if c == "'":
                        if j + 1 < ll and line[j + 1] == "'":
                            j += 2
                            continue
                        local_sq = False
                    j += 1
                    continue
                if local_dq:
                    if c == '"':
                        if j + 1 < ll and line[j + 1] == '"':
                            j += 2
                            continue
                        local_dq = False
                    j += 1
                    continue
                if c == "'":
                    local_sq = True
                    j += 1
                    continue
                if c == '"':
                    local_dq = True
                    j += 1
                    continue
                if c == '!':
                    code_end = j
                    break
                j += 1
            in_sq, in_dq = local_sq, local_dq
            code_part = line[:code_end]
            is_cont = (not in_sq and not in_dq) and code_part.rstrip().endswith('&')
            buf.append(line)
            if is_cont:
                continue
            logical.append('\n'.join(buf))
            buf = []
        if buf:
            logical.append('\n'.join(buf))
        return logical

    # ── Statement Fusion helpers ──────────────────────────────────

    _KEYWORDS = frozenset(w.upper() for w in (
        "program", "end", "module", "submodule", "subroutine", "function",
        "use", "only", "implicit", "none", "integer", "real", "logical",
        "character", "complex", "double", "precision", "type", "class",
        "dimension", "allocatable", "pointer", "target", "intent", "in",
        "out", "inout", "optional", "parameter", "save", "public",
        "private", "protected", "value", "external", "intrinsic",
        "recursive", "pure", "elemental", "impure", "result", "contains",
        "interface", "generic", "operator", "assignment", "if", "then",
        "else", "elseif", "endif", "do", "while", "concurrent", "enddo",
        "exit", "cycle", "select", "case", "selecttype", "default",
        "where", "elsewhere", "endwhere", "forall", "endforall",
        "associate", "endassociate", "block", "endblock", "critical",
        "endcritical", "goto", "continue", "stop", "errorstop", "return",
        "call", "allocate", "deallocate", "nullify", "read", "write",
        "print", "format", "open", "close", "inquire", "rewind",
        "backspace", "endfile", "common", "equivalence", "data",
        "namelist", "entry", "procedure", "abstract", "deferred", "nopass",
        "pass", "bind", "import", "enum", "enumerator", "sequence",
        "volatile", "asynchronous", "codimension", "contiguous", "errmsg",
        "mold", "source", "sync", "lock", "unlock", "team", "event",
        "images", "kind", "len", "blockdata", "final", "extends",
    ))

    _VAR_ASSIGN_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=(?!=)')
    _DECL_RE = re.compile(
        r'^\s*(?:integer|real|logical|character|complex|double\s*precision'
        r'|type\s*\(|class\s*\()[^:]*::\s*(.+)$',
        re.IGNORECASE,
    )
    _DECL_NAME_RE = re.compile(r'([A-Za-z_]\w*)')
    _UNIT_NAME_RE = re.compile(
        r'^\s*(?:[A-Za-z_][\w()*,: \t]*?\s+)?(?:program|module|submodule|subroutine|function)\s+([A-Za-z_]\w*)',
        re.IGNORECASE,
    )
    _CALL_RE = re.compile(r'\bcall\s+([A-Za-z_]\w*)', re.IGNORECASE)
    _IDENT_RE = re.compile(r'\b([A-Za-z_]\w*)\b')
    _NORM_STR = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
    _NORM_NUM = re.compile(r'\b\d+\.?\d*(?:[eEdD][+-]?\d+)?\b')

    _UNIT_OPEN_RE = re.compile(
        r'^(?:[A-Za-z_][\w()*,: \t]*?\s+)?(program|module|submodule|subroutine|function|block\s*data)\b',
        re.IGNORECASE,
    )
    _INNER_OPEN_RE = re.compile(
        r'^(if\s*\(.*\)\s*then\s*$|do\b|select\s*(?:case|type)\b'
        r'|where\s*\(.*\)\s*$|forall\b.*$|type\s*(?!\()(?:,|::|\s|$)'
        r'|interface\b|associate\s*\(|block(?!\s*data)\b|critical\b'
        r'|enum\b|change\s+team\b)',
        re.IGNORECASE,
    )
    _INNER_CLOSE_RE = re.compile(
        r'^end\s*(if|do|select|where|forall|type|interface|associate|block|critical|enum|team)\b',
        re.IGNORECASE,
    )
    _UNIT_CLOSE_RE = re.compile(r'^end\b', re.IGNORECASE)

    @classmethod
    def _classify(cls, code_line: str):
        """Classify a (comment-stripped) line as one of:
        'inner_close', 'unit_close', 'inner_open', 'unit_open', or None."""
        if not code_line:
            return None
        if cls._INNER_CLOSE_RE.match(code_line):
            return 'inner_close'
        if cls._UNIT_CLOSE_RE.match(code_line):
            return 'unit_close'
        if cls._INNER_OPEN_RE.match(code_line):
            return 'inner_open'
        if cls._UNIT_OPEN_RE.match(code_line):
            return 'unit_open'
        return None

    @classmethod
    def _split_statements(cls, code: str) -> List[str]:
        """Split Fortran source into statement units. Each program-unit
        header/footer (PROGRAM/SUBROUTINE/.../END ...) and each ordinary
        body line directly inside a unit (at any unit-nesting depth, e.g.
        a CONTAINS-nested internal subroutine too) becomes its own
        standalone statement; nested control-flow blocks (IF/DO/SELECT/...)
        are kept atomic as a single multi-line unit, mirroring how the
        C/PHP splitters treat compound-statement bodies."""
        logical_lines = cls._join_continuations(code)
        statements: List[str] = []
        current: List[str] = []
        udepth = 0
        idepth = 0

        def flush():
            nonlocal current
            stmt = '\n'.join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []

        for logical in logical_lines:
            first_phys = logical.split('\n', 1)[0]
            code_part = cls._strip_comment(first_phys).strip()
            kind = cls._classify(code_part)

            if kind == 'inner_close':
                # Closing line belongs to whatever inner block has been
                # accumulating in `current` since its matching inner_open —
                # append it there, and once idepth bottoms out the whole
                # nested block (open line through close line) flushes as
                # ONE atomic statement.
                current.append(logical)
                idepth = max(idepth - 1, 0)
                if idepth == 0:
                    flush()
                continue

            if kind == 'unit_close':
                # A program-unit header (PROGRAM/SUBROUTINE/...) is emitted
                # as its own standalone statement below (in 'unit_open'), so
                # its closing END line must also stand alone — never glue it
                # onto whatever inner block happened to flush last.
                if current:
                    flush()
                udepth = max(udepth - 1, 0)
                statements.append(logical)
                continue

            if kind == 'unit_open':
                if current:
                    flush()
                statements.append(logical)
                udepth += 1
                continue

            if kind == 'inner_open':
                current.append(logical)
                idepth += 1
                continue

            # ordinary line
            if udepth >= 1 and idepth == 0:
                if current:
                    flush()
                if logical.strip():
                    statements.append(logical)
            else:
                current.append(logical)

        flush()
        return statements

    def _stmt_defines_uses(self, stmt: str):
        """Return (defines, unit_defines, uses) for a statement unit.
        `unit_defines` (PROGRAM/MODULE/SUBROUTINE/FUNCTION names) is the
        subset eligible for cross-seed dependency edges — plain variable
        assignments/declarations stay seed-local."""
        first_line = self._strip_comment(stmt.split('\n', 1)[0])

        unit_defines = set()
        m = self._UNIT_NAME_RE.match(first_line)
        if m:
            unit_defines.add(m.group(1).upper())

        var_defines = set()
        m = self._VAR_ASSIGN_RE.match(first_line)
        if m and m.group(1).upper() not in self._KEYWORDS:
            var_defines.add(m.group(1).upper())
        m = self._DECL_RE.match(first_line)
        if m:
            for part in m.group(1).split(','):
                nm = self._DECL_NAME_RE.match(part.strip())
                if nm and nm.group(1).upper() not in self._KEYWORDS:
                    var_defines.add(nm.group(1).upper())

        uses = {t.upper() for t in self._IDENT_RE.findall(stmt) if t.upper() not in self._KEYWORDS}

        return (var_defines | unit_defines), unit_defines, uses

    def _normalize_stmt(self, stmt: str) -> str:
        s = self._NORM_STR.sub("'_'", stmt)
        s = self._NORM_NUM.sub('0', s)
        s = self._IDENT_RE.sub(
            lambda m: m.group(0) if m.group(0).upper() in self._KEYWORDS else '_', s)
        return s

    @staticmethod
    def _token_similarity(norm_a: str, norm_b: str) -> float:
        toks_a = set(norm_a.split())
        toks_b = set(norm_b.split())
        if not toks_a and not toks_b:
            return 1.0
        union = toks_a | toks_b
        return len(toks_a & toks_b) / len(union) if union else 0.0

    @classmethod
    def _unit_order_chain(cls, stmts: List[str]) -> List[Tuple[int, int]]:
        """Return (i, j) pairs meaning "statement i must come after
        statement j" for consecutive entries that lie inside the same
        still-open program unit (from its header line through its footer,
        inclusive). Def/use edges alone don't keep a unit's body attached
        to its own header/footer — e.g. a SUBROUTINE's local variable
        declarations don't "use" the subroutine's name — so without this,
        topological sort is free to scatter a unit's body outside its own
        header/footer entirely. This forces the original relative order to
        be preserved for anything still inside an open unit."""
        pairs = []
        depth = 0
        prev_in_unit = None
        for i, s in enumerate(stmts):
            first = cls._strip_comment(s.split('\n', 1)[0]).strip()
            kind = cls._classify(first)
            if kind == 'unit_open':
                if depth > 0 and prev_in_unit is not None:
                    pairs.append((i, prev_in_unit))
                depth += 1
                prev_in_unit = i
                continue
            if kind == 'unit_close':
                if prev_in_unit is not None:
                    pairs.append((i, prev_in_unit))
                depth = max(depth - 1, 0)
                prev_in_unit = i if depth > 0 else None
                continue
            if depth > 0:
                if prev_in_unit is not None:
                    pairs.append((i, prev_in_unit))
                prev_in_unit = i
        return pairs

    def _dependency_graph_interleave(self, stmts_a: List[str], stmts_b: List[str]) -> List[str]:
        tagged = [(s, 'a') for s in stmts_a] + [(s, 'b') for s in stmts_b]
        n = len(tagged)
        if n == 0:
            return []

        info = []
        for stmt, origin in tagged:
            defines, unit_defines, uses = self._stmt_defines_uses(stmt)
            info.append({
                'stmt': stmt, 'origin': origin, 'defines': defines,
                'unit_defines': unit_defines, 'uses': uses,
                'norm': self._normalize_stmt(stmt),
            })

        def_map: Dict[str, List[Tuple[int, str]]] = {}
        for i, inf in enumerate(info):
            for name in inf['defines']:
                def_map.setdefault(name, []).append((i, inf['origin']))

        deps = [set() for _ in range(n)]
        for j, inf_j in enumerate(info):
            for name in inf_j['uses']:
                if name not in def_map:
                    continue
                for di, d_origin in def_map[name]:
                    if di == j:
                        continue
                    if d_origin == inf_j['origin']:
                        deps[j].add(di)
                    elif name in info[di]['unit_defines']:
                        deps[j].add(di)

        # Keep each seed's own program-unit bodies attached to their own
        # header/footer (see _unit_order_chain).
        offset_b = len(stmts_a)
        for i, j in self._unit_order_chain(stmts_a):
            deps[i].add(j)
        for i, j in self._unit_order_chain(stmts_b):
            deps[offset_b + i].add(offset_b + j)

        emitted = [False] * n
        result: List[str] = []
        in_degree = [len(d) for d in deps]
        dependents = [[] for _ in range(n)]
        for j in range(n):
            for di in deps[j]:
                dependents[di].append(j)

        ready = [i for i in range(n) if in_degree[i] == 0]
        last_norm = ""
        while ready:
            if not result:
                pick_idx = random.choice(ready)
            else:
                scored = sorted(
                    ((self._token_similarity(last_norm, info[ri]['norm']), ri) for ri in ready),
                    key=lambda x: -x[0])
                top_k = min(3, len(scored))
                pick_idx = random.choice([s[1] for s in scored[:top_k]])

            ready.remove(pick_idx)
            emitted[pick_idx] = True
            result.append(info[pick_idx]['stmt'])
            last_norm = info[pick_idx]['norm']

            for dep_j in dependents[pick_idx]:
                in_degree[dep_j] -= 1
                if in_degree[dep_j] == 0 and not emitted[dep_j]:
                    ready.append(dep_j)

        for i in range(n):
            if not emitted[i]:
                result.append(info[i]['stmt'])

        return result

    def _replace_random_occurrence_word(self, s: str, old: str, new: str) -> str:
        """Word-boundary-safe, case-insensitive replacement (Fortran is
        case-insensitive, so 'X' and 'x' are the same identifier). Delegates
        to the shared replace_word_occurrences for the actual weighted-
        occurrence-count replacement."""
        return replace_word_occurrences(s, old, new, ignorecase=True)

    def _stmt_cross_replace_variable(self, code: str, vars_a: List[str], vars_b: List[str]) -> str:
        if not vars_a or not vars_b:
            return code
        var_b = random.choice(vars_b)
        var_a = random.choice(vars_a)
        if var_a.upper() == var_b.upper():
            return code
        return self._replace_random_occurrence_word(code, var_b, var_a)

    def _find_body_span(self, stmt: str):
        """For one atomic statement unit (e.g. a whole IF/DO block or a
        whole program unit), return (body_start_line, body_end_line) index
        range (exclusive of the header/footer lines) — or None."""
        logical_lines = self._join_continuations(stmt)
        if len(logical_lines) < 2:
            return None
        first = self._strip_comment(logical_lines[0].split('\n', 1)[0]).strip()
        if self._classify(first) not in ('unit_open', 'inner_open'):
            return None
        depth = 1
        for i in range(1, len(logical_lines)):
            code_part = self._strip_comment(logical_lines[i].split('\n', 1)[0]).strip()
            kind = self._classify(code_part)
            if kind in ('unit_open', 'inner_open'):
                depth += 1
            elif kind in ('unit_close', 'inner_close'):
                depth -= 1
                if depth == 0:
                    if i - 1 < 1:
                        return None
                    return (1, i, logical_lines)
        return None

    # Program-unit headers (PROGRAM/SUBROUTINE/...) are always standalone
    # single-line statements now (see _split_statements), never a multi-line
    # atomic block — only inner control-flow blocks can be injection targets.
    _BLOCK_HEAD_RE = re.compile(
        r'^(if\s*\(.*\)\s*then\s*$|do\b|select\s*(?:case|type)\b'
        r'|where\s*\(.*\)\s*$|forall\b.*$)',
        re.IGNORECASE,
    )

    def _inject_into_block(self, stmts: List[str]) -> List[str]:
        candidates = []
        atomic = []
        for idx, s in enumerate(stmts):
            first = self._strip_comment(s.split('\n', 1)[0]).strip()
            if self._BLOCK_HEAD_RE.match(first):
                span = self._find_body_span(s)
                if span:
                    start, end, lines = span
                    body = '\n'.join(lines[start:end])
                    if len(body.strip()) > 5:
                        candidates.append((idx, start, end, lines))
            elif s.strip() and self._classify(first) not in ('unit_open', 'unit_close'):
                # A lone PROGRAM/SUBROUTINE/... header or its END footer is
                # never a valid donor — injecting one into an unrelated
                # block would open/close a program unit boundary in the
                # middle of that block, breaking the balanced nesting that
                # the rest of the seed (and its own real header/footer)
                # depends on.
                atomic.append((idx, s))

        if not candidates or not atomic:
            return stmts

        target_idx, body_start, body_end, lines = random.choice(candidates)

        donors = [(i, s) for i, s in atomic if i != target_idx]
        if not donors:
            return stmts
        n_inject = min(random.randint(1, 3), len(donors))
        chosen = random.sample(donors, n_inject)
        inject_stmts = [s for _, s in chosen]

        indent = "  "
        for ln in lines[body_start:body_end]:
            stripped = ln.lstrip()
            if stripped:
                indent = ln[:len(ln) - len(stripped)]
                break

        insert_pos = random.randint(body_start, body_end)
        injected = [indent + s.strip() for s in inject_stmts]
        new_lines = lines[:insert_pos] + injected + lines[insert_pos:]

        result = list(stmts)
        result[target_idx] = '\n'.join(new_lines)
        return result

    def _statement_fuse(self, clean1: str, clean2: str,
                         vars1: List[str], vars2: List[str]) -> str:
        stmts_a = self._split_statements(clean1)
        stmts_b = self._split_statements(clean2)

        if not stmts_a and not stmts_b:
            return clean1 + '\n' + clean2

        interleaved = self._dependency_graph_interleave(stmts_a, stmts_b)

        if random.random() < 0.3:
            interleaved = self._inject_into_block(interleaved)

        fused_code = '\n'.join(interleaved)
        fused_code = self._stmt_cross_replace_variable(fused_code, vars1, vars2)
        return fused_code

    # ── Dataflow fusion: insert before the outermost unit's END ──────

    def _lightweight_bridge(self, code1: str, code2: str):
        """Override: Fortran has no valid "append a statement after the
        last program unit" construct, so the bridge assignment goes
        through _insert_before_last_unit_end instead of the base class's
        plain append."""
        vars1 = self._lightweight_vars(code1)
        vars2 = self._lightweight_vars(code2)
        if not vars1 or not vars2:
            return code1, code2
        var1 = random.choice(vars1)
        var2 = random.choice(vars2)
        bridge = self.bridge_var_name
        code1 = self._insert_before_last_unit_end(code1, self.format_assignment(bridge, var1))
        code2 = self._lightweight_replace(code2, var2, bridge)
        return code1, code2

    def interleave_code_blocks(self, code1, code2, dataflow1, dataflow2, extra_flows=None):
        """Override of GenericDataflowStrategy.interleave_code_blocks:
        Fortran has no valid "append a statement after the last program
        unit" construct (unlike C's top-level declaration trick), so the
        bridge assignment is inserted just before code1's outermost unit
        closing END instead of appended at the very end. Bridge-target
        selection goes through _pick_var2 so --dataflow-type-match can
        bias it."""
        if self.lightweight:
            return self._lightweight_bridge(code1, code2)
        if not dataflow1 or not dataflow2:
            return code1, code2

        if extra_flows:
            dataflow1 = dataflow1 + extra_flows

        bridge = self.bridge_var_name

        if random.choice([True, False]):
            try:
                group1 = random.choice(dataflow1)
                var1 = random.choice(group1)
                group2 = random.choice(dataflow2)
                var2 = self._pick_var2(var1, code1, code2, dataflow2, lambda: random.choice(group2))
                bridge_stmt = self.format_assignment(bridge, var1)
                code1 = self._insert_before_last_unit_end(code1, bridge_stmt)
                code2 = self._replace_random_occurrence_word(code2, var2, bridge)
            except IndexError:
                pass
            return code1, code2

        max_df1 = max(dataflow1, key=len) if dataflow1 else []
        max_df2 = max(dataflow2, key=len) if dataflow2 else []

        if max_df1 and max_df2:
            var1 = random.choice(max_df1)
            var2 = self._pick_var2(var1, code1, code2, dataflow2, lambda: random.choice(max_df2))
            bridge_stmt = self.format_assignment(bridge, var1)
            code1 = self._insert_before_last_unit_end(code1, bridge_stmt)
            code2 = self._replace_random_occurrence_word(code2, var2, bridge)

        return code1, code2

    def _insert_before_last_unit_end(self, code: str, statement: str) -> str:
        """Insert `statement` just before the last outermost-unit-closing
        END line in `code` (udepth 1->0 transition). Falls back to
        appending at the end if no complete unit is found (malformed/
        fragment seed — same tolerant behaviour as elsewhere)."""
        logical_lines = self._join_continuations(code)
        udepth = 0
        idepth = 0
        last_close_idx = None
        for i, logical in enumerate(logical_lines):
            first = self._strip_comment(logical.split('\n', 1)[0]).strip()
            kind = self._classify(first)
            if kind == 'inner_close':
                idepth = max(idepth - 1, 0)
            elif kind == 'unit_close':
                udepth = max(udepth - 1, 0)
                if udepth == 0 and idepth == 0:
                    last_close_idx = i
            elif kind == 'unit_open':
                udepth += 1
            elif kind == 'inner_open':
                idepth += 1

        if last_close_idx is None:
            return code + f"\n{statement}\n"

        new_lines = logical_lines[:last_close_idx] + [statement] + logical_lines[last_close_idx:]
        return '\n'.join(new_lines)

    # ── Top-level name-conflict resolution ────────────────────────

    def _extract_top_level_names(self, code: str):
        names = set()
        for logical in self._join_continuations(code):
            first = self._strip_comment(logical.split('\n', 1)[0]).strip()
            m = self._UNIT_NAME_RE.match(first)
            if m:
                names.add(m.group(1).upper())
        return names

    def _resolve_name_conflicts(self, code_a: str, code_b: str) -> str:
        conflicts = self._extract_top_level_names(code_a) & self._extract_top_level_names(code_b)
        if not conflicts:
            return code_b
        result = code_b
        for name in sorted(conflicts, key=len, reverse=True):
            result = re.sub(
                rf'(?<!\w)\b{re.escape(name)}\b(?!\w)',
                name + '_ffl',
                result,
                flags=re.IGNORECASE,
            )
        return result

    # ── Entry points ───────────────────────────────────────────────

    def _build_fused_test(self, parent_a: Seed, parent_b: Seed, mode: str) -> Seed:
        code1 = parent_a.content
        code2 = parent_b.content
        meta1 = parent_a.metadata
        meta2 = parent_b.metadata
        variable1 = meta1.get('variables', [])
        variable2 = meta2.get('variables', [])
        dataflow1 = meta1.get('dataflows', [])
        dataflow2 = meta2.get('dataflows', [])

        if self.mutation:
            code1 = self.mut.mutate(code1)
            code2 = self.mut.mutate(code2)

        code2 = self._resolve_name_conflicts(code1, code2)

        if mode == 'stmt_ab':
            fused = self._statement_fuse(code1, code2, variable1, variable2)
        elif mode == 'stmt_ba':
            fused = self._statement_fuse(code2, code1, variable2, variable1)
        elif mode == 'df_ab':
            new_code1, new_code2 = self.interleave_code_blocks(code1, code2, dataflow1, dataflow2)
            fused = f"{new_code1}\n{new_code2}"
        elif mode == 'df_ba':
            new_code2, new_code1 = self.interleave_code_blocks(code2, code1, dataflow2, dataflow1)
            fused = f"{new_code1}\n{new_code2}"
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ext = meta1.get('extension', '.f90')
        seed_type = meta1.get('type', 'fortran')

        return Seed(content=fused, metadata={
            "parents": [parent_a.id, parent_b.id],
            "type": seed_type,
            "extension": ext,
            "mode": mode,
            "description": f"Fused {parent_a.id} + {parent_b.id} ({mode})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        mode = 'stmt_ab' if self.stmt_fusion else 'df_ab'
        return self._build_fused_test(parent_a, parent_b, mode)

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """Produce both A->B and B->A variants for the active fusion kind."""
        if self.stmt_fusion:
            return [
                self._build_fused_test(parent_a, parent_b, 'stmt_ab'),
                self._build_fused_test(parent_a, parent_b, 'stmt_ba'),
            ]
        return [
            self._build_fused_test(parent_a, parent_b, 'df_ab'),
            self._build_fused_test(parent_a, parent_b, 'df_ba'),
        ]


class FlangStateFusionStrategy(FlangFusionStrategy):
    """
    State fusion for Fortran (core/state_analysis.py): grafts one seed's
    continuation into the other's state at a profiled most complex state
    (CLOSE/DEALLOCATE, an INT/REAL/DBLE conversion, an ERROR STOP/STAT=
    boundary). Fortran has no brace delimiters, so the continuation is
    truncated at the next program-unit boundary (bare END, or a fresh
    PROGRAM/SUBROUTINE/FUNCTION header) using the same keyword-based
    classifier _split_statements already uses, instead of
    core/state_analysis.py's generic brace-balance truncation.
    """

    def _truncate_before_unit_boundary(self, lines: List[str], start_idx: int) -> int:
        depth = 0
        for i in range(start_idx, len(lines)):
            code_part = self._strip_comment(lines[i]).strip()
            kind = self._classify(code_part)
            if depth == 0 and kind in ('unit_close', 'unit_open'):
                return i
            if kind == 'inner_open':
                depth += 1
            elif kind == 'inner_close':
                depth = max(depth - 1, 0)
        return len(lines)

    def _build_state_fused(self, host: Seed, donor: Seed, direction: str) -> Seed:
        from .state_analysis import pick_state_point, graft_continuation, StatePoint

        host_code, donor_code = host.content, donor.content
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)
        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        host_point = pick_state_point(host_code, "flang", self.project_root,
                                       cached=(host.metadata or {}).get("most_complex_states"),
                                       lightweight=self.lightweight)
        donor_point = pick_state_point(donor_code, "flang", self.project_root,
                                        cached=(donor.metadata or {}).get("most_complex_states"),
                                        lightweight=self.lightweight)
        if host_point is None:
            lines = host_code.splitlines()
            host_point = StatePoint(max(len(lines) - 1, 0), "fallback", "", "")

        donor_lines = donor_code.splitlines()
        start_idx = (donor_point.line_idx + 1) if donor_point else 0
        end_idx = self._truncate_before_unit_boundary(donor_lines, start_idx)
        truncated_donor = "\n".join(donor_lines[:end_idx])

        fused = graft_continuation(host_code, truncated_donor, host_point, donor_point)

        ext = host.metadata.get('extension', '.f90')
        seed_type = host.metadata.get('type', 'fortran')

        return Seed(content=fused, metadata={
            "parents": [host.id, donor.id],
            "type": seed_type,
            "extension": ext,
            "mode": f"state_{host_point.category}_{direction}",
            "description": f"State-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_state_fused(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_state_fused(parent_a, parent_b, "ab"),
            self._build_state_fused(parent_b, parent_a, "ba"),
        ]


class FlangDeclarationFusionStrategy(FlangFusionStrategy):
    """
    Declaration fusion for Fortran: injects `EXTENDS(DonorType)` into a
    host derived-type declaration (`TYPE :: Foo` -> `TYPE, EXTENDS(Bar) ::
    Foo`) — Fortran 2003+ type extension, the closest analogue to a
    base-class list. An EXTENDS naming a type that isn't visible, or that
    creates a component/binding conflict, is rejected purely from the
    TYPE declaration itself; no procedure needs to be called.
    """

    _FORTRAN_TYPE_HEADER_RE = re.compile(
        r'(?im)^(?P<indent>[ \t]*)(?P<kw>type)(?P<attrs>\s*,\s*[^:]+)?\s*::\s*(?P<name>[A-Za-z_]\w*)'
    )

    def _inject_extends(self, code: str, donor_name: str):
        matches = list(self._FORTRAN_TYPE_HEADER_RE.finditer(code))
        if not matches:
            return code, False
        m = random.choice(matches)
        if m.group('attrs'):
            if re.search(r'(?i)extends\s*\(', m.group('attrs')):
                return code, False  # Fortran derived types support single inheritance only
            new_code = code[:m.end('attrs')] + f", extends({donor_name})" + code[m.end('attrs'):]
        else:
            new_code = code[:m.end('kw')] + f", extends({donor_name})" + code[m.end('kw'):]
        return new_code, True

    def _build_declaration_fused_test(self, host: Seed, donor: Seed, direction: str) -> Seed:
        host_code, donor_code = host.content, donor.content
        if self.mutation:
            host_code = self.mut.mutate(host_code)
            donor_code = self.mut.mutate(donor_code)
        donor_code = self._resolve_name_conflicts(host_code, donor_code)

        donor_names = [m.group('name') for m in self._FORTRAN_TYPE_HEADER_RE.finditer(donor_code)]
        fused_host, applied = host_code, False
        if donor_names:
            donor_name = random.choice(donor_names)
            fused_host, applied = self._inject_extends(host_code, donor_name)

        fused = f"{donor_code}\n{fused_host}"

        ext = host.metadata.get('extension', '.f90')
        seed_type = host.metadata.get('type', 'fortran')

        return Seed(content=fused, metadata={
            "parents": [host.id, donor.id],
            "type": seed_type,
            "extension": ext,
            "mode": f"decl_{'extends' if applied else 'none'}_{direction}",
            "description": f"Declaration-fused {host.id} <- {donor.id} ({direction})",
        })

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        return self._build_declaration_fused_test(parent_a, parent_b, "ab")

    def fuse_bidirectional(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        return [
            self._build_declaration_fused_test(parent_a, parent_b, "ab"),
            self._build_declaration_fused_test(parent_b, parent_a, "ba"),
        ]


class LFortranFusionStrategy(FlangFusionStrategy):
    """
    LFortran fusion strategy. LFortran and flang are both plain Fortran
    frontends fuzzing off the exact same seed corpus (see
    projects/lfortran/config.yaml) — FlangFusionStrategy's splitter and
    dataflow-bridging logic operates purely on Fortran source syntax with
    no LLVM/flang-specific behavior, so it's reused as-is; only the
    default project_root differs.
    """

    def __init__(self, project_root="projects/lfortran", type_match: bool = False,
                 lightweight: bool = False):
        super().__init__(project_root=project_root, type_match=type_match, lightweight=lightweight)


class LFortranStateFusionStrategy(FlangStateFusionStrategy):
    """State fusion for LFortran — see LFortranFusionStrategy."""

    def __init__(self, project_root="projects/lfortran", lightweight: bool = False):
        super().__init__(project_root=project_root, lightweight=lightweight)


class LFortranDeclarationFusionStrategy(FlangDeclarationFusionStrategy):
    """Declaration fusion for LFortran — see LFortranFusionStrategy."""

    def __init__(self, project_root="projects/lfortran"):
        super().__init__(project_root=project_root)


# ==========================================
# Strategy Factory (Updated)
# ==========================================

def get_strategies(project_name=None, dataflow_fusion=False,
                    struct_fusion=False, declaration_fusion=False, state_fusion=False,
                    dataflow_type_match=False, pre_analysis_enabled=True):
    """
    Build the pool of fusion strategies for `project_name`. Each of
    dataflow_fusion/state_fusion/declaration_fusion independently adds its
    matching strategy (where the project has one) to the pool.
    core/orchestrator.py's process_iteration then *combines* techniques
    from that pool on the same parent pair instead of picking just one:
    one technique always applies, a second layers on top 50% of the
    time, and a third layers on top of that 25% of the time. Pass any
    subset of the flags to restrict the pool (e.g. --dataflow-fusion
    alone always yields a chain of length 1, same as before); pass none
    to get the default, which is now every technique the project
    supports (so combined fusion is on by default) rather than dataflow
    fusion alone.

    dataflow_type_match only affects statically-typed-language dataflow
    strategies (clang, swift, mlir, rust, haskell, flang, lfortran): 90% of
    the time it biases the bridge substitution toward a same-declared-type
    pair (falling back to random/type-agnostic when none exists), 10% of
    the time it deliberately picks a type-mismatched pair — each language's
    own notion of "bridge pair" and "same type" varies (variable-for-
    variable for clang/swift/flang/lfortran, a same-declared-type `let` for
    rust, a same-scalar-type constant group for mlir, an integer-literal
    slot vs. an arbitrary identifier for haskell) but the 90/10 split and
    off-by-default/unchanged-behavior guarantee are consistent across all
    of them. Dynamically-typed languages (cpython, php) always pick the
    bridge pair uniformly at random and ignore this flag — a "same type"
    preference isn't meaningful for them.

    pre_analysis_enabled (--pre-analysis) gates the richer per-seed
    metadata every strategy would otherwise prefer to consume. When False:
      - dataflow fusion falls back to a very lightweight, on-the-fly rule
        for every project: scan each side's variables directly from its
        source text (no parse-time dataflow graph, no dependency-graph
        grouping) and connect one uniformly random pair. (cpython, swift,
        rust, mlir, haskell already work this way regardless of this flag
        — they never depended on cached metadata to begin with; only php/
        clang/flang/lfortran's default dataflow mode normally consumes a
        parse-time dataflow graph, and this flag makes them bypass it.)
      - state fusion falls back to picking any uniformly random line as
        the splice point, instead of core/state_analysis.py's live-
        variable-count analysis (or Haskell's category scan).
      - declaration fusion is disabled outright (not considered feasible
        without --pre-analysis) — a warning is logged if it would
        otherwise have been requested (explicitly or via the "no flags"
        default).
    """
    # Default: if no technique is explicitly requested, enable every
    # technique the project has — process_iteration's chaining is what
    # actually varies how many of them apply to a given child.
    if not dataflow_fusion and not declaration_fusion and not state_fusion and not struct_fusion:
        dataflow_fusion = True
        declaration_fusion = True
        state_fusion = True

    lightweight = not pre_analysis_enabled
    if lightweight and (declaration_fusion or struct_fusion):
        logger.warning(
            "--pre-analysis is not enabled: declaration fusion isn't "
            "feasible without its richer per-seed metadata, so "
            "--declaration-fusion/--struct-fusion is being disabled for "
            "this run. Dataflow fusion falls back to a lightweight, "
            "on-the-fly random-variable-connect rule, and state fusion "
            "falls back to picking any random line, for every project."
        )
        declaration_fusion = False
        struct_fusion = False

    strategies = []

    if project_name == "haskell":
        if os.path.exists("projects/haskell"):
            if dataflow_fusion:
                strategies.append(HaskellFusionStrategy(project_root="projects/haskell",
                                                          type_match=dataflow_type_match,
                                                          lightweight=lightweight))
            if declaration_fusion:
                strategies.append(HaskellDeclarationFusionStrategy(project_root="projects/haskell"))
            if state_fusion:
                strategies.append(HaskellStateFusionStrategy(project_root="projects/haskell",
                                                               lightweight=lightweight))
        return strategies

    if project_name == "php":
        if os.path.exists("projects/php"):
            if dataflow_fusion:
                strategies.append(PHPFusionStrategy(project_root="projects/php",
                                                      lightweight=lightweight))
            if declaration_fusion:
                strategies.append(PHPDeclarationFusionStrategy(project_root="projects/php"))
            if state_fusion:
                strategies.append(PHPStateFusionStrategy(project_root="projects/php",
                                                           lightweight=lightweight))
        return strategies

    if project_name == "clang":
        if os.path.exists("projects/clang"):
            if dataflow_fusion:
                strategies.append(ClangFusionStrategy(project_root="projects/clang",
                                                       type_match=dataflow_type_match,
                                                       lightweight=lightweight))
            if declaration_fusion:
                strategies.append(ClangDeclarationFusionStrategy(project_root="projects/clang"))
            if state_fusion:
                strategies.append(ClangStateFusionStrategy(project_root="projects/clang",
                                                            lightweight=lightweight))
        return strategies

    if project_name == "flang":
        if os.path.exists("projects/flang"):
            if dataflow_fusion:
                strategies.append(FlangFusionStrategy(project_root="projects/flang",
                                                        type_match=dataflow_type_match,
                                                        lightweight=lightweight))
            if declaration_fusion:
                strategies.append(FlangDeclarationFusionStrategy(project_root="projects/flang"))
            if state_fusion:
                strategies.append(FlangStateFusionStrategy(project_root="projects/flang",
                                                             lightweight=lightweight))
        return strategies

    if project_name == "lfortran":
        if os.path.exists("projects/lfortran"):
            if dataflow_fusion:
                strategies.append(LFortranFusionStrategy(project_root="projects/lfortran",
                                                           type_match=dataflow_type_match,
                                                           lightweight=lightweight))
            if declaration_fusion:
                strategies.append(LFortranDeclarationFusionStrategy(project_root="projects/lfortran"))
            if state_fusion:
                strategies.append(LFortranStateFusionStrategy(project_root="projects/lfortran",
                                                                lightweight=lightweight))
        return strategies

    if project_name == "cpython":
        if os.path.exists("projects/cpython"):
            if dataflow_fusion:
                strategies.append(CPythonFusionStrategy(project_root="projects/cpython",
                                                          lightweight=lightweight))
            if declaration_fusion:
                strategies.append(CPythonDeclarationFusionStrategy(project_root="projects/cpython"))
            if state_fusion:
                strategies.append(CPythonStateFusionStrategy(project_root="projects/cpython",
                                                               lightweight=lightweight))
        return strategies

    if project_name == "mlir":
        if os.path.exists("projects/mlir"):
            if dataflow_fusion:
                strategies.append(MLIRFusionStrategy(project_root="projects/mlir",
                                                       type_match=dataflow_type_match,
                                                       lightweight=lightweight))
            if declaration_fusion:
                strategies.append(MLIRDeclarationFusionStrategy(project_root="projects/mlir"))
            if state_fusion:
                strategies.append(MLIRStateFusionStrategy(project_root="projects/mlir",
                                                            lightweight=lightweight))
        return strategies

    if project_name == "rust":
        if os.path.exists("projects/rust"):
            if dataflow_fusion:
                strategies.append(RustFusionStrategy(project_root="projects/rust",
                                                       type_match=dataflow_type_match))
            # --struct-fusion is Rust's original name for this project's
            # declaration-fusion strategy; --declaration-fusion is accepted
            # as an alias so the flag name is consistent across projects.
            # (No state-fusion strategy exists for Rust, so lightweight
            # has nothing to thread into here — Rust's dataflow fusion is
            # already on-the-fly regardless.)
            if struct_fusion or declaration_fusion:
                strategies.append(RustStructFusionStrategy(project_root="projects/rust"))
        return strategies

    if project_name == "swift":
        if os.path.exists("projects/swift"):
            if dataflow_fusion:
                strategies.append(SwiftFusionStrategy(project_root="projects/swift",
                                                       type_match=dataflow_type_match,
                                                       lightweight=lightweight))
            if declaration_fusion:
                strategies.append(SwiftDeclarationFusionStrategy(project_root="projects/swift"))
            if state_fusion:
                strategies.append(SwiftStateFusionStrategy(project_root="projects/swift",
                                                            lightweight=lightweight))
        return strategies

    # Fallback / Legacy behavior — only for the "no project specified" case;
    # an unrecognized (or removed) project_name falls through to here too if
    # we don't guard it, and would otherwise silently get an arbitrary mix
    # of other languages' strategies instead of the empty pool the caller
    # should see (main.py fails fast on an empty pool).
    if project_name is None:
        if os.path.exists("projects/php"):
            strategies.append(PHPFusionStrategy(project_root="projects/php"))
        if os.path.exists("projects/cpython"):
            strategies.append(CPythonFusionStrategy(project_root="projects/cpython"))
        if os.path.exists("projects/mlir"):
            strategies.append(MLIRFusionStrategy(project_root="projects/mlir"))
        if os.path.exists("projects/rust"):
            strategies.append(RustFusionStrategy(project_root="projects/rust"))
    return strategies
