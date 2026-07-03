import abc
import uuid
import random
import sqlite3
import os
import re
import io
import tokenize
import keyword
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
from .mutation import BaseMutator, PHPMutator, CPythonMutator, RustMutator, WGSLMutator, GoMutator, LeanMutator, JSMutator, CangjeMutator

@dataclass
class Seed:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    metadata: dict = field(default_factory=dict)

class FusionStrategy(abc.ABC):
    @abc.abstractmethod
    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        pass

# ==========================================
# Helpers
# ==========================================

def replace_random_occurrence(s, old, new):
    """
    Simple string replacement for non-sensitive contexts.
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

    random_pos = random.choice(positions)
    return s[:random_pos] + new + s[random_pos + len(old):]

def replace_random_occurrence_indented(s, old, new):
    """
    Context-aware replacement that respects indentation.
    Vital for Python to prevent IndentationError.
    """
    matches = list(re.finditer(re.escape(old), s))
    if not matches:
        return s

    match = random.choice(matches)
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

    return s[:start] + replacement + s[end:]

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
    Replaces a variable in B with 'replacement', handling indentation context.
    """
    if not name: return src_text, False
    
    patt = re.compile(rf'(?<!\.)\b{re.escape(name)}\b(?!\s*\.)')
    
    matches = list(patt.finditer(src_text))
    if not matches:
        return src_text, False
    
    match = random.choice(matches)
    start, end = match.span()
    
    line_start = src_text.rfind('\n', 0, start) + 1
    indent_match = re.match(r'^(\s*)', src_text[line_start:start])
    indent = indent_match.group(1) if indent_match else ""
    
    final_repl = replacement
    if '\n' in replacement:
        lines = replacement.split('\n')
        final_repl = lines[0] + '\n' + '\n'.join([indent + l for l in lines[1:]])
    
    new_text = src_text[:start] + final_repl + src_text[end:]
    return new_text, True

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
    def __init__(self, mutator: BaseMutator = None):
        self.mut = mutator if mutator else BaseMutator()
        self.bridge_var_name = "fusion"

    @abc.abstractmethod
    def format_assignment(self, lhs: str, rhs: str) -> str:
        pass

    def interleave_code_blocks(self, code1, code2, dataflow1, dataflow2, extra_flows=None):
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
    def __init__(self, project_root="projects/php"):
        super().__init__(mutator=PHPMutator())
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

    def fuse_all(self, parent_a: Seed, parent_b: Seed) -> List[Seed]:
        """Produce all four fusion variants for one pair."""
        return [
            self._build_fused_test(parent_a, parent_b, 'stmt_ab'),
            self._build_fused_test(parent_a, parent_b, 'stmt_ba'),
            self._build_fused_test(parent_a, parent_b, 'df_ab'),
            self._build_fused_test(parent_a, parent_b, 'df_ba'),
        ]

# ==========================================
# CPython Specific Fusion Strategy
# ==========================================

class CPythonFusionStrategy(FusionStrategy):
    def __init__(self, project_root="projects/cpython"):
        self.project_root = project_root
        self.mutation = True
        self.bridge_var_name = "fusion"
        self.mut = CPythonMutator()

    def _are_types_compatible(self, types_a, types_b):
        if not types_a.isdisjoint(types_b): return True
        numerics = {'int', 'float', 'bool', 'complex'}
        if not types_a.isdisjoint(numerics) and not types_b.isdisjoint(numerics): return True
        collections = {'list', 'tuple', 'set', 'frozenset', 'dict', 'range'}
        if not types_a.isdisjoint(collections) and not types_b.isdisjoint(collections): return True
        return False

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
        meta1 = parent_a.metadata
        meta2 = parent_b.metadata
        types1 = meta1.get('dynamic_types') or {}
        types2 = meta2.get('dynamic_types') or {}

        if self.mutation:
            sa = self.mut.mutate(sa)
            sb = self.mut.mutate(sb)

        a_body, a_imports = self._extract_imports_and_body(sa)
        b_body, b_imports = self._extract_imports_and_body(sb)
        all_imports = sorted(list(set(a_imports + b_imports)))

        # ── A-side candidates: prefer runtime-live vars over syntactic ones ──
        # live_vars (from dry-run) = variables confirmed alive at program end:
        # not deleted, not inside a failed try block, not conditionally unset.
        # Fall back to syntactic collect when dry-run data is absent.
        syntactic_a  = collect_top_level_assigned_vars(a_body)
        live_a       = meta1.get('live_vars') or []
        if live_a:
            # Intersection: syntactically assigned AND confirmed alive.
            # This filters out variables that were assigned but then del'd or
            # only assigned inside branches that didn't execute.
            live_set = set(live_a)
            a_candidates = [v for v in syntactic_a if v in live_set] or live_a
        else:
            a_candidates = syntactic_a

        # ── B-side candidates: prefer undefined references over all bare vars ──
        # undefined_refs = identifiers B uses but never defines/imports.
        # Replacing one with 'fusion' satisfies a NameError rather than
        # disrupting B's internal dataflow — maximally valid bridge targets.
        # Fall back to collect_bare_vars when dry-run data is absent.
        undef_b      = meta2.get('undefined_refs') or []
        b_candidates = undef_b if undef_b else collect_bare_vars(b_body)

        a_var   = None
        b_choice = None

        if a_candidates and b_candidates:
            matching_pairs  = []
            wildcard_pairs  = []
            for va in a_candidates:
                t_a = set(types1.get(va, []))
                t_a.discard('function'); t_a.discard('builtin_function_or_method'); t_a.discard('type')
                is_a_wildcard = (len(t_a) == 0)
                for vb in b_candidates:
                    t_b = set(types2.get(vb, []))
                    t_b.discard('function'); t_b.discard('builtin_function_or_method'); t_b.discard('type')
                    is_b_wildcard = (len(t_b) == 0)
                    if is_a_wildcard or is_b_wildcard:
                        wildcard_pairs.append((va, vb))
                        continue
                    if self._are_types_compatible(t_a, t_b):
                        matching_pairs.append((va, vb))

            if matching_pairs:
                a_var, b_choice = random.choice(matching_pairs)
            elif wildcard_pairs:
                a_var, b_choice = random.choice(wildcard_pairs)

        if not a_var and a_candidates:   a_var    = a_candidates[-1]
        if not b_choice and b_candidates: b_choice = random.choice(b_candidates)

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

# ==========================================
# Swift Specific Fusion Strategy
# ==========================================

class SwiftFusionStrategy(FusionStrategy):
    """
    Swift-Specific Fusion Strategy.
    Ensures correct structure by strictly ordering imports and bodies,
    and attempting simple type-safe bridging of values.
    """
    def __init__(self, project_root="projects/swift"):
        self.project_root = project_root
        self.bridge_var_name = "fusion"

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
        Attempts to find a literal in 'code' that matches the 'source_type' and replace it with 'source_var_name'.
        This avoids re-definition errors by modifying existing logic flow instead.
        """
        lines = code.splitlines()
        new_lines = []
        replaced = False
        
        # Regexes for literals
        int_lit = r'(?<![a-zA-Z0-9_])\d+(?![a-zA-Z0-9_])'
        str_lit = r'"[^"]*"'
        bool_lit = r'\b(true|false)\b'
        
        target_regex = None
        if source_type == "Int": target_regex = int_lit
        elif source_type == "String": target_regex = str_lit
        elif source_type == "Bool": target_regex = bool_lit

        if not target_regex:
            return code, False

        for line in lines:
            if not replaced and re.search(target_regex, line):
                # Avoid replacing inside comments or imports
                if "//" in line or "import " in line:
                    new_lines.append(line)
                    continue
                    
                # Replace FIRST occurrence of the literal with the variable name
                # This bridges the dataflow: B uses A's variable instead of its own constant
                new_line = re.sub(target_regex, source_var_name, line, count=1)
                new_lines.append(new_line)
                replaced = True
            else:
                new_lines.append(line)
        
        return "\n".join(new_lines), replaced

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
        # Try to replace a compatible literal in B with the bridge variable
        final_b_body = b_body
        if bridge_var:
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

# ==========================================
# MLIR Specific Fusion Strategy
# ==========================================

class MLIRFusionStrategy(FusionStrategy):
    def __init__(self, project_root="projects/mlir"):
        self.project_root = project_root

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
        """
        all_consts = a_consts + b_consts
        if not all_consts:
            return ""

        # Try to find a pair with the same scalar type
        ty = None
        same = []
        for candidate_ty in dict.fromkeys(c["ty"] for c in all_consts):
            group = [c for c in all_consts if c["ty"] == candidate_ty]
            # Only use simple scalar types to stay safe across all passes
            if len(group) >= 2 and re.fullmatch(r'[iuf]\d+|index', candidate_ty):
                ty = candidate_ty
                same = group
                break

        if not same:
            return ""

        picks = random.sample(same, min(4, len(same)))
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


# ==========================================
# Go Specific Fusion Strategy
# ==========================================

class GoFusionStrategy(FusionStrategy):
    """
    Go-Specific Fusion Strategy.
    1. Strips package declarations and hoists merged imports.
    2. Renames 'func main()' in each parent to a unique name.
    3. Creates a new 'func main()' that calls both renamed mains.
    4. Appends phase-directed bug primitives targeting the Go compiler.
    """

    # Go keywords — never treat as variable names
    _GO_KW = frozenset({
        'break', 'case', 'chan', 'const', 'continue', 'default', 'defer',
        'else', 'fallthrough', 'for', 'func', 'go', 'goto', 'if', 'import',
        'interface', 'map', 'package', 'range', 'return', 'select', 'struct',
        'switch', 'type', 'var',
        # predeclared identifiers
        'nil', 'true', 'false', 'iota', 'any', 'comparable',
        'int', 'int8', 'int16', 'int32', 'int64',
        'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr',
        'float32', 'float64', 'complex64', 'complex128',
        'bool', 'byte', 'rune', 'string', 'error',
        'append', 'cap', 'close', 'complex', 'copy', 'delete',
        'imag', 'len', 'make', 'new', 'panic', 'print', 'println',
        'real', 'recover',
    })

    def __init__(self, project_root="projects/go"):
        self.project_root = project_root
        self.mut = GoMutator()

    def _process_seed(self, code, uid):
        """
        - Strips //go:build and // +build constraints.
        - Strips the package declaration.
        - Extracts all import paths (block and single form).
        - Renames 'func main()' to 'ffl_main_<uid>'.
        Returns (imports: set[str], body: str, new_main_name: str|None)
        """
        # Strip build directives
        code = re.sub(r'//go:build[^\n]*\n', '', code)
        code = re.sub(r'// \+build[^\n]*\n', '', code)

        # Strip package declaration
        code = re.sub(r'^\s*package\s+\w+[ \t]*\n?', '', code, count=1, flags=re.MULTILINE)

        # Extract block imports: import ( "a"\n "b" )
        imports: set = set()
        block_re = re.compile(r'import\s*\(([^)]*)\)', re.S)
        for m in block_re.finditer(code):
            for path in re.findall(r'(?:[A-Za-z_]\w*\s+|_\s+)?"([^"]+)"', m.group(1)):
                imports.add(path)
        code = block_re.sub('', code)

        # Extract single imports: import "pkg" or import alias "pkg"
        single_re = re.compile(r'import\s+(?:[A-Za-z_]\w*\s+|_\s+)?"([^"]+)"')
        for m in single_re.finditer(code):
            imports.add(m.group(1))
        code = single_re.sub('', code)

        # Rename func main() -> ffl_main_<uid>
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid)
        new_main = None
        main_re = r'(func\s+)main(\s*\(\s*\))'
        if re.search(main_re, code):
            new_main = f"ffl_main_{uid_safe}"
            code = re.sub(main_re, rf'\1{new_main}\2', code, count=1)

        return imports, code.strip(), new_main

    # Top-level declaration keywords whose names can collide
    _TOPLEVEL_RE = re.compile(
        r'^(?:func|type|var|const)\s+([A-Za-z_]\w*)',
        re.MULTILINE
    )

    def _rename_collisions(self, body_a: str, body_b: str, uid_b: str) -> str:
        """
        Find top-level identifiers (func/type/var/const) declared in both
        body_a and body_b, and rename the colliding ones in body_b by
        appending a uid suffix — prevents 'X redeclared in this block' errors.
        """
        names_a = {m.group(1) for m in self._TOPLEVEL_RE.finditer(body_a)}
        names_b = [m.group(1) for m in self._TOPLEVEL_RE.finditer(body_b)]
        collisions = {n for n in names_b if n in names_a} - self._GO_KW
        if not collisions:
            return body_b
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid_b)
        # Rename longest names first to avoid partial matches
        for name in sorted(collisions, key=len, reverse=True):
            new_name = f"{name}_b{uid_safe}"
            body_b = re.sub(r'\b' + re.escape(name) + r'\b', new_name, body_b)
        return body_b

    def _bug_primitives(self):
        """
        Phase-directed bug primitives for the Go compiler.
        Each primitive targets a distinct compiler phase.
        All types/functions are prefixed _ffl_ to avoid clashing with seed code.
        """

        # P1: Escape analysis — stack vs heap allocation decisions
        p1 = '''
// P1: Escape analysis & heap allocation
type _ffl_p1Node struct{ val int; next *_ffl_p1Node }

func _ffl_p1() {
\tn := &_ffl_p1Node{val: 42}
\tn.next = &_ffl_p1Node{val: n.val + 1}
\t// Closure capturing n forces it to escape to heap
\tfn := func() *_ffl_p1Node { return n }
\t_ = fn()
\t// Slice: compiler decides stack vs heap based on size
\ts := make([]int, 1<<4)
\tfor i := range s { s[i] = i * i }
\t_ = s
}'''

        # P2: Inliner budget — mix of inlineable and non-inlineable calls
        p2 = '''
// P2: Inliner budget stress
//go:noinline
func _ffl_p2Heavy(x int) int {
\tsum := 0
\tfor i := 0; i < x; i++ { sum += i * i }
\treturn sum
}

func _ffl_p2Trivial(x int) int { return x ^ (x >> 1) }

func _ffl_p2Chain(a, b int) int {
\treturn _ffl_p2Trivial(_ffl_p2Trivial(a) + _ffl_p2Trivial(b))
}

func _ffl_p2() {
\t_ = _ffl_p2Heavy(10)
\t_ = _ffl_p2Chain(3, 7)
\t// Indirect call via function variable — devirtualization opportunity
\tvar fn func(int) int = _ffl_p2Trivial
\t_ = fn(99)
}'''

        # P3: Interface dispatch, itab caching, and type assertion
        p3 = '''
// P3: Interface dispatch & type assertion stress
type _ffl_p3Iface interface {
\tVal() int
\tTag() string
}
type _ffl_p3A struct{ v int }
func (a _ffl_p3A) Val() int    { return a.v }
func (a _ffl_p3A) Tag() string { return "A" }
type _ffl_p3B struct{ v int }
func (b _ffl_p3B) Val() int    { return b.v * 2 }
func (b _ffl_p3B) Tag() string { return "B" }

func _ffl_p3() {
\tvals := []_ffl_p3Iface{_ffl_p3A{1}, _ffl_p3B{2}, _ffl_p3A{3}, _ffl_p3B{4}}
\tfor _, v := range vals {
\t\t_ = v.Val()
\t\t_ = v.Tag()
\t\t// Type switch: stresses type assertion & itab lookup
\t\tswitch x := v.(type) {
\t\tcase _ffl_p3A:
\t\t\t_ = x.v
\t\tcase _ffl_p3B:
\t\t\t_ = x.v
\t\t}
\t}
\t// Empty interface boxing/unboxing
\tvar ei interface{} = _ffl_p3A{42}
\tif a, ok := ei.(_ffl_p3A); ok { _ = a }
}'''

        # P4: Stack growth, defer ordering, and panic/recover
        p4 = '''
// P4: Stack growth, defer ordering & panic/recover
func _ffl_p4Recurse(n int) int {
\tif n <= 0 { return 0 }
\tdefer func() { recover() }()
\treturn n + _ffl_p4Recurse(n-1)
}

func _ffl_p4() {
\t_ = _ffl_p4Recurse(64)
\t// Defer ordering under panic+recover: defers run LIFO
\tfunc() {
\t\tdefer func() { recover() }()
\t\tfor i := 0; i < 4; i++ {
\t\t\ti := i // capture loop variable
\t\t\tdefer func() { _ = i }()
\t\t}
\t\tpanic("ffl: recover test")
\t}()
}'''

        # P5: Bounds check elimination (BCE) & range loop optimizations
        p5 = '''
// P5: Bounds check elimination & range loop optimization
func _ffl_p5() {
\ts := []int{10, 20, 30, 40, 50}
\tn := len(s)
\t// BCE: compiler proves i < n, eliminates runtime bounds check
\tfor i := 0; i < n; i++ { _ = s[i] }
\tsum := 0
\tfor _, v := range s { sum += v }
\t_ = sum
\t// 2D slice: nested BCE
\tmatrix := [][]int{{1, 2}, {3, 4}, {5, 6}}
\tfor i := range matrix {
\t\tfor j := range matrix[i] { _ = matrix[i][j] }
\t}
}'''

        # P6: Constant folding, overflow, and dead code elimination
        p6 = '''
// P6: Constant folding, integer overflow & DCE
const (
\t_ffl_p6C1 int64  = 1<<63 - 1  // max int64
\t_ffl_p6C2 int64  = -1 << 63   // min int64
\t_ffl_p6C3 uint64 = 1<<64 - 1  // max uint64
)

func _ffl_p6() {
\t_ = _ffl_p6C1 + _ffl_p6C2    // intentional signed overflow
\t_ = _ffl_p6C1 ^ _ffl_p6C2    // XOR of boundary constants
\t_ = _ffl_p6C3 >> 1            // large constant right-shift
\tconst f = 1.0 / 3.0           // irrational constant folding
\t_ = f * 3.0
\t// Dead code: compiler should eliminate unreachable branch via DCE
\tif false { panic("ffl: DCE unreachable") }
\tconst alwaysTrue = (1 == 1)
\tif !alwaysTrue { panic("ffl: const-false branch") }
}'''

        # P7: Generics monomorphization (Go 1.18+)
        p7 = '''
// P7: Generics monomorphization stress (Go 1.18+)
func _ffl_p7Min[T interface{ ~int | ~int64 | ~float64 }](a, b T) T {
\tif a < b { return a }
\treturn b
}

func _ffl_p7Map[T, U any](s []T, f func(T) U) []U {
\tout := make([]U, len(s))
\tfor i, v := range s { out[i] = f(v) }
\treturn out
}

type _ffl_p7Stack[T any] struct{ items []T }
func (st *_ffl_p7Stack[T]) Push(v T)       { st.items = append(st.items, v) }
func (st *_ffl_p7Stack[T]) Pop() (T, bool) {
\tvar zero T
\tif len(st.items) == 0 { return zero, false }
\tv := st.items[len(st.items)-1]
\tst.items = st.items[:len(st.items)-1]
\treturn v, true
}

func _ffl_p7() {
\t// Multiple concrete instantiations — stresses monomorphization
\t_ = _ffl_p7Min(3, 7)
\t_ = _ffl_p7Min(3.14, 2.72)
\t_ = _ffl_p7Map([]int{1, 2, 3}, func(x int) string { return fmt.Sprint(x) })
\tvar st _ffl_p7Stack[int]
\tst.Push(1); st.Push(2); st.Push(3)
\tfor { if _, ok := st.Pop(); !ok { break } }
}'''

        # P8: Goroutine, channel, and select scheduling
        p8 = '''
// P8: Goroutine, channel & select stress
func _ffl_p8() {
\ttype _key struct{ a, b int }
\tm := map[_key]string{_key{1, 2}: "ab", _key{3, 4}: "cd"}
\tfor k, v := range m { _ = k; _ = v }
\t// Buffered channel: goroutine send + main recv
\tch := make(chan int, 1)
\tgo func() { ch <- 42 }()
\t_ = <-ch
\t// Select with default: non-blocking probe
\tselect {
\tcase v := <-ch:
\t\t_ = v
\tdefault:
\t}
}'''

        # P9: reflect & unsafe interaction with the runtime
        p9 = '''
// P9: reflect & interface representation stress
import "reflect"

func _ffl_p9() {
\tvals := []interface{}{42, "hello", 3.14, true, []int{1, 2, 3}, map[string]int{"a": 1}}
\tfor _, v := range vals {
\t\trv := reflect.ValueOf(v)
\t\t_ = rv.Kind()
\t\t_ = rv.Type()
\t\t// Stringer probe: if the type implements Stringer, call it
\t\tif s, ok := v.(interface{ String() string }); ok { _ = s.String() }
\t}
\t// Struct field iteration via reflection
\ttype _ffl_s struct { X int; Y string }
\trv := reflect.ValueOf(_ffl_s{X: 1, Y: "ffl"})
\tfor i := 0; i < rv.NumField(); i++ {
\t\t_ = rv.Field(i)
\t\t_ = rv.Type().Field(i).Name
\t}
}'''

        # p9 uses a bare 'import "reflect"' at function scope (invalid Go);
        # omit until reflect is added to the merged import block.
        safe_phases = [p1, p2, p3, p4, p5, p6, p7, p8]
        # selected = random.sample(safe_phases, k=random.choice([2, 3]))
        return random.choice(safe_phases)
        # return "\n".join(selected)

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        imports_a, body_a, main_a = self._process_seed(code_a, parent_a.id)
        imports_b, body_b, main_b = self._process_seed(code_b, parent_b.id)

        # Rename top-level declarations in B that collide with A
        body_b = self._rename_collisions(body_a, body_b, parent_b.id)

        # Merge imports; always include "fmt" (used in main and p7)
        all_imports = sorted((imports_a | imports_b) | {"fmt"})
        import_block = "import (\n" + "".join(f'\t"{p}"\n' for p in all_imports) + ")"

        # New main: call both renamed mains + use fmt to satisfy import checker
        main_lines = ["func main() {"]
        if main_a:
            main_lines.append(f"\t{main_a}()")
        if main_b:
            main_lines.append(f"\t{main_b}()")
        main_lines.append('\tfmt.Println("FFL Fusion Done")')
        main_lines.append("}")
        new_main = "\n".join(main_lines)

        bug_prims = self._bug_primitives()

        parts = [
            "package main",
            "",
            import_block,
            "",
            f"// === Seed A: {parent_a.id} ===",
            body_a,
            "",
            f"// === Seed B: {parent_b.id} ===",
            body_b,
            "",
            "// === FFL Bug Primitives ===",
            bug_prims,
            "",
            "// === FFL Fused Main ===",
            new_main,
        ]
        final_content = "\n".join(parts)

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "go",
                "description": f"Fused {parent_a.id} + {parent_b.id}"
            }
        )


class RustFusionStrategy(FusionStrategy):
    """
    Rust-Specific Fusion Strategy.
    1. Extracts 'use' statements and hoists them.
    2. Renames 'fn main' in parents to unique names.
    3. Creates a new 'fn main' that calls the parent mains.
    4. Injects random dataflow bridging between variables if possible (naive regex).
    """
    def __init__(self, project_root="projects/rust"):
        self.project_root = project_root
        self.mut = RustMutator()
        self.bridge_var_name = "fusion_var"

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

    def _pick_bridge_pair(self, meta_a: dict, meta_b: dict):
        """
        Pick (var_a, var_b, needs_clone) from dryrun metadata.

        Priority:
          1. Matching Copy types in both seeds         → direct assignment, no clone
          2. Matching Clone-able types in both seeds   → assignment + .clone()
          3. Any Copy var from A × any var in B        → direct (type-unsafe but may work)
          4. None → fall back to no bridge
        """
        var_types_a: dict = meta_a.get("var_types", {})
        var_types_b: dict = meta_b.get("var_types", {})
        prim_a: list      = meta_a.get("primitive_vars", [])
        clone_a: list     = meta_a.get("cloneable_vars", [])

        if not var_types_a or not var_types_b:
            return None, None, False

        # Candidates on the B side: any variable that has a declared type
        b_vars = list(var_types_b.keys())
        if not b_vars:
            return None, None, False

        # --- Pass 1: Copy-type pairs ---
        for va in prim_a:
            ta = var_types_a.get(va, "")
            for vb in b_vars:
                tb = var_types_b.get(vb, "")
                if self._types_compatible(ta, tb):
                    return va, vb, False        # direct: let fusion_var = va;

        # --- Pass 2: Clone-able pairs ---
        for va in clone_a:
            if va in prim_a:
                continue                        # already tried
            ta = var_types_a.get(va, "")
            for vb in b_vars:
                tb = var_types_b.get(vb, "")
                if self._types_compatible(ta, tb):
                    return va, vb, True         # clone:  let fusion_var = va.clone();

        # --- Pass 3: Any Copy var from A × first B var (type mismatch may occur,
        #             but Copy types are at least free of borrow issues) ---
        if prim_a:
            return prim_a[0], random.choice(b_vars), False

        return None, None, False

    # ------------------------------------------------------------------

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        meta_a = parent_a.metadata or {}
        meta_b = parent_b.metadata or {}

        # Process A and B: separate crate attrs, use lines, and body
        attrs_a, uses_a, body_a, main_a = self._process_seed(code_a, parent_a.id)
        attrs_b, uses_b, body_b, main_b = self._process_seed(code_b, parent_b.id)

        # Merge: crate-level attrs MUST come first (dedup), then use lines
        all_attrs = list(dict.fromkeys(attrs_a + attrs_b))
        all_uses = sorted(set(uses_a + uses_b))

        # ── Type-aware bridge ──────────────────────────────────────────
        # Attempt to inject a bridge statement that passes a value from
        # seed A into the fn main of seed B, using metadata collected
        # during the dry-run. This keeps the fused program ownership-safe.
        bridge_stmt = ""
        var_a, var_b, needs_clone = self._pick_bridge_pair(meta_a, meta_b)
        if var_a and var_b:
            rhs = f"{var_a}.clone()" if needs_clone else var_a
            bridge_stmt = (
                f"\n    // FFL bridge: {var_a} ({'cloned' if needs_clone else 'copy'}) "
                f"→ {var_b}\n"
                f"    let {self.bridge_var_name} = {rhs};\n"
                f"    let _ = {self.bridge_var_name}; // suppress unused warning\n"
            )

        # ── Assemble new fn main ───────────────────────────────────────
        new_main_body = ["fn main() {"]
        if main_a:
            new_main_body.append(f"    {main_a}();")
        if bridge_stmt:
            new_main_body.append(bridge_stmt)
        if main_b:
            new_main_body.append(f"    {main_b}();")
        new_main_body.append('    println!("FFL Fusion Done");')
        new_main_body.append("}")

        # Assemble: #![...] attrs → use lines → bodies → new main
        parts = []
        if all_attrs:
            parts.append("\n".join(all_attrs))
        if all_uses:
            parts.append("\n".join(all_uses))
        parts.append(f"// Seed A\n{body_a}")
        parts.append(f"// Seed B\n{body_b}")
        parts.append("\n".join(new_main_body))
        final_content = "\n\n".join(parts)

        return Seed(
            content=final_content,
            metadata={
                "parents":     [parent_a.id, parent_b.id],
                "type":        "rust",
                "description": f"Fused {parent_a.id} + {parent_b.id}",
                "bridge_var":  var_a,
            }
        )

# ==========================================
# Strategy Factory (Updated)
# ==========================================

# ==========================================
# WGSL Specific Helpers
# ==========================================

WGSL_FN_DEF = re.compile(
    r'((?:@\w+(?:\([^)]*\))?\s*)*)'   # attributes like @vertex, @compute @workgroup_size(N)
    r'fn\s+([A-Za-z_]\w*)\s*\(',      # fn name(
    re.S
)

WGSL_STRUCT_DEF = re.compile(r'struct\s+([A-Za-z_]\w*)\s*\{', re.S)

WGSL_CONST_DEF = re.compile(
    r'(const\s+)([A-Za-z_]\w*)\s*:\s*([^=;]+?)\s*=\s*([^;]+);',
    re.S
)

def wgsl_extract_functions(src):
    """Extract function names and their full text blocks from WGSL source."""
    funcs = []
    matches = list(WGSL_FN_DEF.finditer(src))
    for i, m in enumerate(matches):
        name = m.group(2)
        start = m.start()
        # Find the matching closing brace
        brace_start = src.find('{', m.end())
        if brace_start == -1:
            continue
        depth = 0
        end = brace_start
        for j in range(brace_start, len(src)):
            if src[j] == '{':
                depth += 1
            elif src[j] == '}':
                depth -= 1
                if depth == 0:
                    end = j + 1
                    break
        funcs.append({"name": name, "start": start, "end": end, "text": src[start:end]})
    return funcs

def wgsl_extract_globals(src, func_spans):
    """Extract top-level declarations that are NOT inside functions."""
    covered = set()
    for f in func_spans:
        for i in range(f["start"], f["end"]):
            covered.add(i)

    global_lines = []
    pos = 0
    for line in src.splitlines(keepends=True):
        line_start = pos
        pos += len(line)
        if line_start not in covered:
            global_lines.append(line)

    return "".join(global_lines)

def wgsl_rename_functions(src, suffix):
    """Rename all user-defined functions by appending suffix."""
    funcs = wgsl_extract_functions(src)
    func_names = [f["name"] for f in funcs]

    result = src
    for name in sorted(func_names, key=len, reverse=True):
        new_name = f"{name}_{suffix}"
        result = re.sub(r'\b' + re.escape(name) + r'\b', new_name, result)

    return result, func_names

def wgsl_extract_consts(src):
    """Extract const declarations with name, type, and value."""
    consts = []
    for m in WGSL_CONST_DEF.finditer(src):
        consts.append({
            "name": m.group(2),
            "type": m.group(3).strip(),
            "value": m.group(4).strip(),
            "span": m.span(),
            "line": m.group(0)
        })
    return consts


class WGSLFusionStrategy(FusionStrategy):
    """
    WGSL-Specific Fusion Strategy for naga.
    1. Extracts globals (struct, const, var) and functions from both parents.
    2. Renames functions to avoid collision.
    3. Cross-pollinates constants between parents.
    4. Concatenates into a single WGSL module.
    5. Applies WGSL-specific mutations.
    """
    def __init__(self, project_root="projects/naga"):
        self.project_root = project_root
        self.mut = WGSLMutator()

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        # 1. Apply mutations
        code_a = self.mut.mutate(code_a)
        code_b = self.mut.mutate(code_b)

        # 2. Rename functions in both parents
        a_suffix = parent_a.id.replace("-", "")[:6]
        b_suffix = parent_b.id.replace("-", "")[:6]
        code_a, fns_a = wgsl_rename_functions(code_a, "a" + a_suffix)
        code_b, fns_b = wgsl_rename_functions(code_b, "b" + b_suffix)

        # 3. Extract functions and globals
        funcs_a = wgsl_extract_functions(code_a)
        funcs_b = wgsl_extract_functions(code_b)
        globals_a = wgsl_extract_globals(code_a, funcs_a)
        globals_b = wgsl_extract_globals(code_b, funcs_b)

        # 4. Rename structs in B to avoid collision with A
        structs_a = set(m.group(1) for m in WGSL_STRUCT_DEF.finditer(globals_a))
        structs_b = set(m.group(1) for m in WGSL_STRUCT_DEF.finditer(globals_b))
        colliding = structs_a & structs_b
        for sname in sorted(colliding, key=len, reverse=True):
            new_name = sname + "_b" + b_suffix
            globals_b = re.sub(r'\b' + re.escape(sname) + r'\b', new_name, globals_b)
            for idx, f in enumerate(funcs_b):
                funcs_b[idx]["text"] = re.sub(r'\b' + re.escape(sname) + r'\b', new_name, f["text"])

        # 4b. Rename var/let/override declarations in B that collide with A
        # Pattern: optional @group/@binding attributes before var<...> name
        WGSL_VAR_DECL = re.compile(
            r'(?:@\w+(?:\([^)]*\))?\s*)*'   # optional attributes
            r'\bvar(?:<[^>]*>)?\s+'          # var<...>
            r'([A-Za-z_]\w*)\s*:',           # name :
            re.S
        )
        vars_a = set(m.group(1) for m in WGSL_VAR_DECL.finditer(globals_a))
        vars_b = set(m.group(1) for m in WGSL_VAR_DECL.finditer(globals_b))
        colliding_vars = vars_a & vars_b
        for vname in sorted(colliding_vars, key=len, reverse=True):
            new_name = vname + "_b" + b_suffix
            globals_b = re.sub(r'\b' + re.escape(vname) + r'\b', new_name, globals_b)
            for idx, f in enumerate(funcs_b):
                funcs_b[idx]["text"] = re.sub(r'\b' + re.escape(vname) + r'\b', new_name, f["text"])

        # 4c. Deduplicate @group/@binding pairs: if B has the same (group, binding) as A, bump B's binding
        WGSL_BINDING = re.compile(r'@group\s*\((\d+)\)\s*@binding\s*\((\d+)\)')
        bindings_a = set((m.group(1), m.group(2)) for m in WGSL_BINDING.finditer(globals_a))

        def bump_binding(m):
            g, b = m.group(1), m.group(2)
            if (g, b) in bindings_a:
                return f"@group({g}) @binding({int(b) + 100})"
            return m.group(0)

        globals_b = WGSL_BINDING.sub(bump_binding, globals_b)

        # 4d. Handle WGSL module-level directives from B.
        # - `diagnostic`: remove from B (A's version is used; duplicates illegal)
        # - `requires`: remove from B (same reason)
        # - `enable`: keep unique ones from B (different extensions may be needed)
        WGSL_DIAG_REQ = re.compile(
            r'^\s*(?:diagnostic\s*\([^)]*\)|requires\s+[^;]+)\s*;\s*\n?',
            re.MULTILINE
        )
        WGSL_ENABLE = re.compile(
            r'^\s*(enable\s+[^;]+;)\s*\n?',
            re.MULTILINE
        )
        # Collect enable directives already in A
        enables_a = set(m.group(1).strip() for m in WGSL_ENABLE.finditer(globals_a))
        # Collect enable directives unique to B (need to be hoisted to module top)
        enables_b_unique = [
            m.group(1).strip() for m in WGSL_ENABLE.finditer(globals_b)
            if m.group(1).strip() not in enables_a
        ]
        # Remove diagnostic/requires from B — replace with newline to avoid text merging
        globals_b = WGSL_DIAG_REQ.sub("\n", globals_b)
        # Remove all enable from B (unique ones will be hoisted in assembly step)
        globals_b = WGSL_ENABLE.sub("", globals_b)

        # 5. Build bridge: pick a const from A and inject into B
        consts_a = wgsl_extract_consts(globals_a)
        bridge_code = ""
        if consts_a:
            picked = random.choice(consts_a)
            bridge_name = "ffl_bridge_" + str(random.randrange(10**6))
            bridge_code = "const " + bridge_name + ": " + picked["type"] + " = " + picked["value"] + ";\n"

        # 6. Assemble
        # WGSL requires: enable/diagnostic/requires MUST come before any declarations.
        # Order: comment → B's unique enables (hoisted) → globals_a → globals_b → bridge → functions
        parts = []
        parts.append("// FFL Fused: " + parent_a.id + " + " + parent_b.id)
        parts.append("")
        # Hoist B's unique enable directives to top (before any declarations)
        if enables_b_unique:
            parts.append("// === Extra Enables (from B) ===")
            parts.append("\n".join(enables_b_unique))
            parts.append("")
        parts.append("// === Globals A ===")
        parts.append(globals_a.strip())
        parts.append("")
        parts.append("// === Globals B ===")
        parts.append(globals_b.strip())
        parts.append("")
        if bridge_code:
            parts.append("// === Bridge ===")
            parts.append(bridge_code)
        parts.append("// === Functions A ===")
        for f in funcs_a:
            parts.append(f["text"])
            parts.append("")
        parts.append("// === Functions B ===")
        for f in funcs_b:
            parts.append(f["text"])
            parts.append("")

        final_content = "\n".join(parts)

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "wgsl",
                "description": "Fused " + parent_a.id + " + " + parent_b.id
            }
        )


class LeanFusionStrategy(FusionStrategy):
    """
    Lean 4 Specific Fusion Strategy.
    1. Extracts and deduplicates 'import' statements from both parents.
    2. Renames top-level declarations in B that collide with A (appends uid suffix).
    3. Concatenates both bodies under section comments.
    4. Injects phase-directed bug primitives targeting the Lean 4 elaborator:
       universe polymorphism, type class synthesis, inductive types, tactics,
       dependent types, mutual recursion, monad do-notation, and #eval stress.
    """

    # Built-in names never renamed during collision resolution
    _LEAN_KW = frozenset({
        'def', 'theorem', 'lemma', 'structure', 'class', 'instance',
        'inductive', 'abbrev', 'noncomputable', 'private', 'protected',
        'namespace', 'section', 'end', 'open', 'variable', 'universe',
        'import', 'export', 'macro', 'elab', 'syntax', 'notation',
        'if', 'then', 'else', 'match', 'with', 'fun', 'let', 'in',
        'have', 'show', 'from', 'by', 'do', 'return', 'pure', 'bind',
        'where', 'deriving', 'extends', 'mut', 'for',
        'Nat', 'Int', 'Float', 'Bool', 'String', 'List', 'Array',
        'Option', 'Result', 'IO', 'Prop', 'Type', 'Sort', 'True', 'False',
        'And', 'Or', 'Not', 'Iff', 'Eq', 'Ne', 'HEq',
        'true', 'false', 'none', 'some', 'rfl',
        'id', 'Function', 'Fin', 'Char', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
    })

    # Matches top-level named declarations
    _TOPLEVEL_RE = re.compile(
        r'^[ \t]*(?:(?:private|protected|noncomputable|partial)\s+)*'
        r'(?:def|theorem|lemma|structure|class|inductive|abbrev|instance)\s+'
        r'([A-Za-z_]\w*)',
        re.MULTILINE,
    )

    def __init__(self, project_root="projects/lean"):
        self.project_root = project_root
        self.mut = LeanMutator()

    def _process_seed(self, code: str):
        """
        Split a Lean 4 file into (imports: set[str], body: str).

        All 'import <Module.Path>' lines are extracted regardless of their
        position in the file (lean4 test files sometimes place them mid-file
        as negative test cases, and Lean requires imports to be first).
        Trailing/leading blank lines are stripped from the returned body.
        """
        imports: set = set()
        body_lines = []
        for line in code.splitlines():
            # Match all Lean 4 import variants:
            #   import Foo.Bar
            #   public import Foo.Bar   (re-export, package builds only)
            #   meta import Foo.Bar     (meta-level import, package builds only)
            # 'public'/'meta' modifiers are stripped — standalone lean files
            # don't support package-level import modifiers.
            m = re.match(r'^\s*(?:public\s+|meta\s+)?import\s+([\w][\w.]*)\s*$', line)
            if m:
                imports.add(m.group(1))
            else:
                body_lines.append(line)
        return imports, "\n".join(body_lines).strip()

    def _rename_collisions(self, body_a: str, body_b: str, uid_b: str) -> str:
        """
        Find top-level identifiers declared in both body_a and body_b,
        and rename the colliding ones in body_b by appending a uid suffix.
        Prevents 'already declared' elaboration errors.
        """
        names_a = {m.group(1) for m in self._TOPLEVEL_RE.finditer(body_a)}
        collisions = (
            {m.group(1) for m in self._TOPLEVEL_RE.finditer(body_b)}
            & names_a
            - self._LEAN_KW
        )
        if not collisions:
            return body_b
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid_b)
        for name in sorted(collisions, key=len, reverse=True):
            body_b = re.sub(r'\b' + re.escape(name) + r'\b', f"{name}_b{uid_safe}", body_b)
        return body_b

    def _bug_primitives(self) -> str:
        """
        Phase-directed bug primitives targeting distinct Lean 4 elaboration phases.
        All names are prefixed _ffl_ to avoid clashing with seed declarations.
        """
        # P1: Universe polymorphism — stresses universe unification and level inference
        p1 = '''\
-- P1: Universe polymorphism
universe _ffl_u _ffl_v
def _ffl_p1Id.{_ffl_u} {α : Sort _ffl_u} (a : α) : α := a
def _ffl_p1Comp.{_ffl_u _ffl_v _ffl_w}
    {α : Sort _ffl_u} {β : Sort _ffl_v} {γ : Sort _ffl_w}
    (f : β → γ) (g : α → β) : α → γ := fun a => f (g a)
theorem _ffl_p1IdIdem.{_ffl_u} {α : Sort _ffl_u} (a : α) :
    _ffl_p1Id (_ffl_p1Id a) = a := rfl'''

        # P2: Type class synthesis — stresses instance search and unification
        p2 = '''\
-- P2: Type class synthesis
class _ffl_p2Combine (α : Type*) where
  empty  : α
  merge  : α → α → α
instance : _ffl_p2Combine Nat  where empty := 0; merge a b := a + b
instance : _ffl_p2Combine Int  where empty := 0; merge a b := a + b
instance : _ffl_p2Combine Bool where empty := false; merge a b := a || b
def _ffl_p2Use [_ffl_p2Combine α] (x y : α) : α := _ffl_p2Combine.merge x y
#eval _ffl_p2Use (3 : Nat) 5'''

        # P3: Inductive types with structural recursion — stresses WF checker + pattern match
        p3 = '''\
-- P3: Inductive types & structural recursion
inductive _ffl_p3Tree (α : Type*) where
  | leaf : _ffl_p3Tree α
  | node : _ffl_p3Tree α → α → _ffl_p3Tree α → _ffl_p3Tree α
def _ffl_p3Size {α : Type*} : _ffl_p3Tree α → Nat
  | .leaf       => 0
  | .node l _ r => 1 + _ffl_p3Size l + _ffl_p3Size r
def _ffl_p3Mirror {α : Type*} : _ffl_p3Tree α → _ffl_p3Tree α
  | .leaf       => .leaf
  | .node l v r => .node (_ffl_p3Mirror r) v (_ffl_p3Mirror l)
theorem _ffl_p3MirrorSize {α : Type*} (t : _ffl_p3Tree α) :
    _ffl_p3Size (_ffl_p3Mirror t) = _ffl_p3Size t := by
  induction t with
  | leaf => rfl
  | node l v r ihl ihr =>
      simp [_ffl_p3Size, _ffl_p3Mirror, ihl, ihr, Nat.add_comm]'''

        # P4: Tactic elaboration — stresses omega, ring, decide, simp, cases
        p4 = '''\
-- P4: Tactic elaboration
theorem _ffl_p4NatAdd (n : Nat) : n + 0 = n          := by omega
theorem _ffl_p4IntComm (n m : Int) : n + m = m + n   := by ring
theorem _ffl_p4BoolAnd (b : Bool) : (b && true) = b  := by cases b <;> rfl
theorem _ffl_p4Decide  : (2 : Nat) + 2 = 4           := by decide
theorem _ffl_p4NatMul  (n : Nat) : n * 0 = 0         := by omega'''

        # P5: Dependent types with Fin / Subtype — stresses definitional equality
        p5 = '''\
-- P5: Dependent types (Fin, Subtype)
def _ffl_p5Head {n : Nat} (f : Fin (n + 1) → Nat) : Nat :=
  f ⟨0, Nat.zero_lt_succ n⟩
def _ffl_p5Sum (n : Nat) (f : Fin n → Nat) : Nat :=
  (List.finRange n).foldl (fun acc i => acc + f i) 0
theorem _ffl_p5FinVal {n : Nat} (h : n < 10) :
    (⟨n, h⟩ : Fin 10).val = n := rfl'''

        # P6: Mutual recursion — stresses the mutual block elaborator
        p6 = '''\
-- P6: Mutual structural recursion
mutual
  def _ffl_p6Even : Nat → Bool
    | 0     => true
    | n + 1 => _ffl_p6Odd n
  def _ffl_p6Odd : Nat → Bool
    | 0     => false
    | n + 1 => _ffl_p6Even n
end
theorem _ffl_p6EvenZero : _ffl_p6Even 0 = true  := rfl
theorem _ffl_p6OddZero  : _ffl_p6Odd  0 = false := rfl'''

        # P7: Monad / do-notation — stresses bind/pure desugaring and type inference
        p7 = '''\
-- P7: Monad / do-notation
def _ffl_p7SafeDiv (a b : Int) : Option Int :=
  if b = 0 then none else some (a / b)
def _ffl_p7Chain (x y z : Int) : Option Int := do
  let a ← _ffl_p7SafeDiv x y
  let b ← _ffl_p7SafeDiv a z
  pure (b + 1)
#eval _ffl_p7Chain 100 5 2    -- some 11
#eval _ffl_p7Chain 100 0 2    -- none'''

        # P8: #eval computation — stresses kernel reduction and IO monad
        p8 = '''\
-- P8: #eval computation stress
#eval (List.range 16).map (· * ·)
#eval (List.range 10).foldl (· + ·) 0
#eval "FFL".toList.map Char.toNat
#eval Nat.gcd 1071 462'''

        safe_phases = [p1, p2, p3, p4, p5, p6, p7, p8]
        selected = random.sample(safe_phases, k=random.choice([2, 3]))
        return "\n\n".join(selected)

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        imports_a, body_a = self._process_seed(code_a)
        imports_b, body_b = self._process_seed(code_b)

        # Rename colliding top-level names in B
        body_b = self._rename_collisions(body_a, body_b, parent_b.id)

        # Merge imports (dedup, sorted).
        # Lean 4 requires ALL import lines to appear before any other content.
        all_imports = sorted(imports_a | imports_b)
        import_block = "\n".join(f"import {i}" for i in all_imports)

        bug_prims = self._bug_primitives()

        # Build sections; skip empty import block so we don't emit a blank
        # first line (lean parser is sensitive to leading whitespace/newlines).
        sections = []
        if import_block:
            sections.append(import_block)
        sections.append(f"-- === Seed A: {parent_a.id} ===\n{body_a}")
        sections.append(f"-- === Seed B: {parent_b.id} ===\n{body_b}")
        sections.append(f"-- === FFL Bug Primitives ===\n{bug_prims}")
        final_content = "\n\n".join(sections)

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "lean",
                "description": f"Fused {parent_a.id} + {parent_b.id}",
            },
        )


class V8FusionStrategy(FusionStrategy):
    """
    V8 / JavaScript Fusion Strategy.
    1. Wraps each parent in an IIFE so their top-level variables don't collide.
    2. Renames colliding top-level function declarations in B.
    3. Injects phase-directed bug primitives that target V8's JIT tiers
       (Ignition, Maglev, Turbofan), GC, typed arrays, Proxy, and Wasm.
    """

    _JS_KW = frozenset({
        'break', 'case', 'catch', 'class', 'const', 'continue', 'debugger',
        'default', 'delete', 'do', 'else', 'export', 'extends', 'false',
        'finally', 'for', 'function', 'if', 'import', 'in', 'instanceof',
        'let', 'new', 'null', 'return', 'static', 'super', 'switch', 'this',
        'throw', 'true', 'try', 'typeof', 'undefined', 'var', 'void', 'while',
        'with', 'yield', 'async', 'await', 'of', 'from', 'get', 'set',
        # Global objects
        'Object', 'Array', 'Function', 'Number', 'String', 'Boolean',
        'Symbol', 'BigInt', 'Math', 'Date', 'RegExp', 'Error', 'Map', 'Set',
        'WeakMap', 'WeakSet', 'Promise', 'Proxy', 'Reflect', 'JSON',
        'ArrayBuffer', 'DataView', 'Int8Array', 'Uint8Array', 'Uint8ClampedArray',
        'Int16Array', 'Uint16Array', 'Int32Array', 'Uint32Array',
        'Float32Array', 'Float64Array', 'BigInt64Array', 'BigUint64Array',
        'WebAssembly', 'console', 'globalThis', 'Infinity', 'NaN',
        # d8 helpers
        'print', 'gc', 'readline', 'load', 'quit', 'version',
    })

    _FUNC_DECL_RE = re.compile(
        r'^(?:async\s+)?function\s*\*?\s*([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(',
        re.MULTILINE,
    )
    _CLASS_DECL_RE = re.compile(
        r'^class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
        re.MULTILINE,
    )

    def __init__(self, project_root="projects/v8"):
        self.project_root = project_root
        self.mut = JSMutator()

    def _top_level_names(self, code: str) -> set:
        names = set()
        for m in self._FUNC_DECL_RE.finditer(code):
            names.add(m.group(1))
        for m in self._CLASS_DECL_RE.finditer(code):
            names.add(m.group(1))
        return names - self._JS_KW

    def _rename_collisions(self, body_a: str, body_b: str, uid_b: str) -> str:
        names_a = self._top_level_names(body_a)
        collisions = self._top_level_names(body_b) & names_a
        if not collisions:
            return body_b
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid_b)
        for name in sorted(collisions, key=len, reverse=True):
            body_b = re.sub(r'\b' + re.escape(name) + r'\b', f"{name}_b{uid_safe}", body_b)
        return body_b

    def _wrap_iife(self, code: str, label: str) -> str:
        return f"// === {label} ===\n(function() {{\n{code}\n}})();"

    def _bug_primitives(self) -> str:
        # P1: Typed array boundary — stresses bounds-check elimination in Turbofan
        p1 = '''\
// P1: Typed array boundary stress
(function _ffl_p1() {
  const _ffl_buf = new ArrayBuffer(16);
  const _ffl_i32 = new Int32Array(_ffl_buf);
  const _ffl_f64 = new Float64Array(_ffl_buf);
  _ffl_i32[0] = 0x7fffffff;
  _ffl_i32[1] = -1;
  _ffl_f64[0] = Infinity;
  _ffl_f64[1] = -0;
  for (let _ffl_i = 0; _ffl_i < 4; _ffl_i++) {
    _ffl_i32[_ffl_i] = _ffl_i32[_ffl_i] | 0;
  }
})();'''

        # P2: Prototype chain manipulation — stresses IC and map transitions
        p2 = '''\
// P2: Prototype chain / hidden-class stress
(function _ffl_p2() {
  function _ffl_C() { this.x = 1; }
  _ffl_C.prototype.m = function() { return this.x; };
  const _ffl_o = new _ffl_C();
  Object.defineProperty(_ffl_o, 'x', { value: 42, writable: false });
  for (let _ffl_i = 0; _ffl_i < 100; _ffl_i++) {
    _ffl_o.y = _ffl_i;  // polymorphic property addition
  }
  delete _ffl_o.y;
})();'''

        # P3: Deoptimization — force OSR + deopt cycle
        p3 = '''\
// P3: Deoptimization (OSR + type feedback pollution)
(function _ffl_p3() {
  function _ffl_add(a, b) { return a + b; }
  // Warm up with ints
  for (let _ffl_i = 0; _ffl_i < 10000; _ffl_i++) _ffl_add(_ffl_i, 1);
  // Deopt with string
  _ffl_add("deopt", 0);
  // Warm up with floats
  for (let _ffl_i = 0; _ffl_i < 10000; _ffl_i++) _ffl_add(_ffl_i * 0.5, 1.5);
})();'''

        # P4: Proxy traps — stresses IC and megamorphic lookups
        p4 = '''\
// P4: Proxy traps
(function _ffl_p4() {
  const _ffl_target = { x: 1, y: 2 };
  const _ffl_proxy = new Proxy(_ffl_target, {
    get(t, k) { return k in t ? t[k] * 2 : undefined; },
    set(t, k, v) { t[k] = v + 1; return true; },
    has(t, k) { return k in t; },
  });
  _ffl_proxy.x;
  _ffl_proxy.z = 10;
  'x' in _ffl_proxy;
})();'''

        # P5: Generator / async stress — stresses bytecode resume points
        p5 = '''\
// P5: Generator + async stress
(async function _ffl_p5() {
  function* _ffl_gen(n) {
    for (let _ffl_i = 0; _ffl_i < n; _ffl_i++) yield _ffl_i * _ffl_i;
  }
  for (const _ffl_v of _ffl_gen(8)) { void _ffl_v; }
  const _ffl_p = await Promise.resolve(42);
  void _ffl_p;
})().catch(() => {});'''

        # P6: Map / Set operations — stresses hash table internals
        p6 = '''\
// P6: Map / Set hash table stress
(function _ffl_p6() {
  const _ffl_m = new Map();
  const _ffl_s = new Set();
  for (let _ffl_i = 0; _ffl_i < 64; _ffl_i++) {
    _ffl_m.set(_ffl_i, _ffl_i * _ffl_i);
    _ffl_s.add(_ffl_i % 7);
  }
  _ffl_m.delete(0);
  _ffl_m.forEach((v, k) => { void (v + k); });
})();'''

        phases = [p1, p2, p3, p4, p5, p6]
        selected = random.sample(phases, k=random.choice([2, 3]))
        return "\n\n".join(selected)

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        code_a = parent_a.content
        code_b = parent_b.content

        if self.mut:
            code_a = self.mut.mutate(code_a)
            code_b = self.mut.mutate(code_b)

        # Rename top-level collisions in B to avoid redeclaration errors
        code_b = self._rename_collisions(code_a, code_b, parent_b.id)

        # Wrap each parent in an IIFE for variable isolation
        block_a = self._wrap_iife(code_a, f"Seed A: {parent_a.id}")
        block_b = self._wrap_iife(code_b, f"Seed B: {parent_b.id}")

        bug_prims = self._bug_primitives()
        bug_block = f"// === FFL Bug Primitives ===\n{bug_prims}"

        final_content = "\n\n".join([block_a, block_b, bug_block])

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type": "javascript",
                "description": f"Fused {parent_a.id} + {parent_b.id}",
            },
        )


# ==========================================
# Cangjie Fusion Strategy
# ==========================================

class CangjieFusionStrategy(FusionStrategy):
    """
    Cangjie-specific fusion strategy.
    1. Hoists and deduplicates 'import' statements from both parents.
    2. Renames top-level func/class/struct/enum declarations in B that
       collide with A (appends a uid suffix).
    3. Renames 'main' in both parents to unique names and emits a new
       combined main() that calls them both.
    4. Injects a simple bridge variable between the two bodies.
    """

    _IMPORT_RE = re.compile(r'^\s*import\s+([\w.]+)', re.MULTILINE)

    # Matches top-level named declarations
    _TOPLEVEL_RE = re.compile(
        r'^[ \t]*(?:(?:public|private|protected|open|abstract|override)\s+)*'
        r'(?:func|class|struct|enum|interface|extend)\s+'
        r'([A-Za-z_]\w*)',
        re.MULTILINE,
    )

    _CJ_KW = frozenset({
        'main', 'func', 'class', 'struct', 'enum', 'interface', 'extend',
        'let', 'var', 'if', 'else', 'for', 'while', 'do', 'match',
        'return', 'break', 'continue', 'import', 'package',
        'true', 'false', 'this', 'super', 'init', 'new',
        'Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64',
        'Float32', 'Float64', 'Bool', 'String', 'Char', 'Unit', 'Nothing',
        'Array', 'ArrayList', 'HashMap', 'Option', 'Some', 'None',
        'println', 'print',
    })

    def __init__(self, project_root="projects/cangjie"):
        self.project_root = project_root
        self.mut = CangjeMutator()

    def _process_seed(self, code: str, uid: str):
        """Extract imports and body; rename main() → main_<uid>() if present."""
        imports: set = set()
        body_lines = []
        for line in code.splitlines():
            m = self._IMPORT_RE.match(line)
            if m:
                imports.add(m.group(1))
            else:
                body_lines.append(line)
        body = "\n".join(body_lines).strip()

        main_name = None
        # Match bare `main()` (entry point, no `func` keyword) at top level
        main_re = re.compile(r'^([ \t]*)main\s*\(\s*\)', re.MULTILINE)
        if main_re.search(body):
            new_name = f"main_{uid}"
            # Rename and add `func` so it becomes a regular callable function
            body = main_re.sub(lambda m: f"{m.group(1)}func {new_name}()", body, count=1)
            main_name = new_name

        return imports, body, main_name

    def _rename_collisions(self, body_a: str, body_b: str, uid_b: str) -> str:
        names_a = {m.group(1) for m in self._TOPLEVEL_RE.finditer(body_a)}
        collisions = (
            {m.group(1) for m in self._TOPLEVEL_RE.finditer(body_b)}
            & names_a
            - self._CJ_KW
        )
        if not collisions:
            return body_b
        uid_safe = re.sub(r'[^a-zA-Z0-9]', '_', uid_b)
        for name in sorted(collisions, key=len, reverse=True):
            # Use a negative lookbehind for '.' so we don't rename inside
            # qualified names like pkg.Name or extend pkg.Name { ... }
            pattern = r'(?<!\.)\b' + re.escape(name) + r'\b'
            body_b = re.sub(pattern, f"{name}_b{uid_safe}", body_b)
        return body_b

    def fuse(self, parent_a: Seed, parent_b: Seed) -> Seed:
        uid_a = re.sub(r'[^a-zA-Z0-9]', '_', parent_a.id)
        uid_b = re.sub(r'[^a-zA-Z0-9]', '_', parent_b.id)

        imports_a, body_a, main_a = self._process_seed(parent_a.content, uid_a)
        imports_b, body_b, main_b = self._process_seed(parent_b.content, uid_b)

        body_b = self._rename_collisions(body_a, body_b, uid_b)

        all_imports = sorted(imports_a | imports_b)
        import_block = "\n".join(f"import {i}" for i in all_imports)

        # Build combined main() — no bridge: local vars inside main_xxx() are
        # not in scope here, so referencing them would cause undeclared-identifier errors.
        main_body_lines = ["main() {"]
        if main_a:
            main_body_lines.append(f"    {main_a}()")
        if main_b:
            main_body_lines.append(f"    {main_b}()")
        main_body_lines.append("}")
        new_main = "\n".join(main_body_lines)

        parts = []
        if import_block:
            parts.append(import_block)
        parts.append(f"// Seed A: {parent_a.id}\n{body_a}")
        parts.append(f"// Seed B: {parent_b.id}\n{body_b}")
        parts.append(new_main)
        final_content = "\n\n".join(parts)

        # Apply Cangjie-aware mutations to the fused result (50 % chance)
        if random.random() < 0.5:
            final_content = self.mut.mutate(final_content)

        return Seed(
            content=final_content,
            metadata={
                "parents": [parent_a.id, parent_b.id],
                "type":    "cangjie",
                "description": f"Fused {parent_a.id} + {parent_b.id}",
            },
        )


# ==========================================
# Strategy Factory (Updated)
# ==========================================

def get_strategies(project_name=None, stmt_fusion=False, dataflow_fusion=False, all_fusion=False):
    # Default: if neither flag given, enable dataflow fusion only
    if not stmt_fusion and not dataflow_fusion and not all_fusion:
        dataflow_fusion = True

    strategies = []

    if project_name == "cangjie":
        if os.path.exists("projects/cangjie"):
            strategies.append(CangjieFusionStrategy(project_root="projects/cangjie"))
        return strategies

    if project_name == "php":
        if os.path.exists("projects/php"):
            s = PHPFusionStrategy(project_root="projects/php")
            s.stmt_fusion = stmt_fusion
            s.dataflow_fusion = dataflow_fusion
            s.all_fusion = all_fusion
            strategies.append(s)
        return strategies

    if project_name == "cpython":
        if os.path.exists("projects/cpython"):
            strategies.append(CPythonFusionStrategy(project_root="projects/cpython"))
        return strategies

    if project_name == "mlir":
        if os.path.exists("projects/mlir"):
            strategies.append(MLIRFusionStrategy(project_root="projects/mlir"))
        return strategies

    if project_name == "rust":
        if os.path.exists("projects/rust"):
            strategies.append(RustFusionStrategy(project_root="projects/rust"))
        return strategies

    if project_name == "go":
        if os.path.exists("projects/go"):
            strategies.append(GoFusionStrategy(project_root="projects/go"))
        return strategies

    if project_name == "naga":
        if os.path.exists("projects/naga"):
            strategies.append(WGSLFusionStrategy(project_root="projects/naga"))
        return strategies
    
    if project_name == "wgslc":
        if os.path.exists("projects/wgslc"):
            strategies.append(WGSLFusionStrategy(project_root="projects/wgslc"))
        return strategies

    if project_name == "lean":
        if os.path.exists("projects/lean"):
            strategies.append(LeanFusionStrategy(project_root="projects/lean"))
        return strategies

    if project_name == "v8":
        if os.path.exists("projects/v8"):
            strategies.append(V8FusionStrategy(project_root="projects/v8"))
        return strategies

    if project_name == "swift":
        if os.path.exists("projects/swift"):
            strategies.append(SwiftFusionStrategy(project_root="projects/swift"))
        return strategies

    # Fallback / Legacy behavior (Load all found)
    if os.path.exists("projects/php"):
        strategies.append(PHPFusionStrategy(project_root="projects/php"))
    if os.path.exists("projects/cpython"):
        strategies.append(CPythonFusionStrategy(project_root="projects/cpython"))
    if os.path.exists("projects/mlir"):
        strategies.append(MLIRFusionStrategy(project_root="projects/mlir"))
    if os.path.exists("projects/rust"):
        strategies.append(RustFusionStrategy(project_root="projects/rust"))
    if os.path.exists("projects/go"):
        strategies.append(GoFusionStrategy(project_root="projects/go"))
    return strategies
