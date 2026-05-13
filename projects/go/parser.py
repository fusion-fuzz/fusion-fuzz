import os
import sqlite3
import json
import re

# Reject seeds that will never compile standalone under 'go run'
_RELATIVE_IMPORT_RE = re.compile(r'"\.\.?/')
_INTERNAL_PKG_RE    = re.compile(r'"/internal/|/internal"')
_INTERNAL_CMD_RE    = re.compile(r'"cmd/[^"]+/internal')

# ---------------------------------------------------------------------------
# Zero-value call generation
# ---------------------------------------------------------------------------

_BASIC_TYPE_ZEROS = {
    'int': '0', 'int8': '0', 'int16': '0', 'int32': '0', 'int64': '0',
    'uint': '0', 'uint8': '0', 'uint16': '0', 'uint32': '0', 'uint64': '0',
    'uintptr': '0', 'byte': '0', 'rune': '0',
    'float32': '0.0', 'float64': '0.0',
    'complex64': '0', 'complex128': '0',
    'bool': 'false', 'string': '""',
}

# Tokens that are type names or keywords — not valid parameter names.
_TYPE_KEYWORDS = frozenset({
    'func', 'map', 'chan', 'interface', 'struct', 'var', 'type', 'any',
    'int', 'int8', 'int16', 'int32', 'int64',
    'uint', 'uint8', 'uint16', 'uint32', 'uint64', 'uintptr',
    'float32', 'float64', 'complex64', 'complex128',
    'bool', 'byte', 'rune', 'string', 'error',
})


def _zero_value_expr(type_str):
    """Return a Go zero-value expression for type_str, or None if unknown."""
    t = type_str.strip()
    if t in _BASIC_TYPE_ZEROS:
        return _BASIC_TYPE_ZEROS[t]
    if t.startswith(('[]', '*', 'map[', 'chan ', 'chan<-', '<-chan')):
        return 'nil'
    if t.startswith('['):
        # Array: [N]T{} is a valid zero-value composite literal
        return f'{t}{{}}'
    if re.match(r'^[A-Za-z_]\w*$', t):
        # Named type — try composite literal (works for structs; zero value for others)
        return f'{t}{{}}'
    return None  # Complex/unparseable type


def _split_top_comma(s):
    """Split s by top-level commas (ignoring commas inside brackets)."""
    groups, depth, current = [], 0, ''
    for c in s:
        if c in '([{':
            depth += 1; current += c
        elif c in ')]}':
            depth -= 1; current += c
        elif c == ',' and depth == 0:
            groups.append(current.strip()); current = ''
        else:
            current += c
    if current.strip():
        groups.append(current.strip())
    return groups


def _parse_param_types(params_str):
    """
    Parse a Go parameter list (content between outer parens).
    Returns a list of type strings (one per parameter), or None if unparseable.

    Handles Go's grouped-name syntax: "a, b uint64" means two params of type uint64.
    After a comma-split, a bare identifier with no following type token looks ahead
    to the next group's type (the standard Go grouping rule).
    """
    params_str = params_str.strip()
    if not params_str:
        return []
    # Bail on complex inline types we can't easily handle
    if any(kw in params_str for kw in ('func(', 'interface{', 'struct{')):
        return None

    # First pass: classify each comma-separated group as (name_count, type_or_None)
    raw = []
    for group in _split_top_comma(params_str):
        group = group.strip()
        if not group:
            continue
        if group.startswith('...'):
            return None  # variadic — skip

        m = re.match(r'^((?:[a-zA-Z_]\w*\s*,\s*)*[a-zA-Z_]\w*)\s+(.+)$', group, re.DOTALL)
        if m:
            name_tokens = [n.strip() for n in m.group(1).split(',')]
            potential_type = m.group(2).strip()
            if any(n in _TYPE_KEYWORDS for n in name_tokens):
                # "names" are actually type keywords → whole group is an unnamed type
                raw.append((1, group))
            else:
                raw.append((len(name_tokens), potential_type))
        else:
            # No space found: either a bare name (grouped type follows) or unnamed type
            if re.match(r'^[a-zA-Z_]\w*$', group) and group not in _TYPE_KEYWORDS:
                raw.append((1, None))   # bare name — type comes from a later group
            else:
                raw.append((1, group))  # unnamed type parameter

    # Second pass: fill in None types by looking ahead (Go's grouped-name rule).
    # e.g. ['a'→None, 'b uint64'→'uint64'] → both get 'uint64'
    filled = []
    for i, (count, type_str) in enumerate(raw):
        if type_str is None:
            look_type = next((raw[j][1] for j in range(i + 1, len(raw))
                              if raw[j][1] is not None), None)
            if look_type is None:
                return None  # Cannot determine type
            filled.append((count, look_type))
        else:
            filled.append((count, type_str))

    result = []
    for count, type_str in filled:
        result.extend([type_str] * count)
    return result


def _build_main_body(group_funcs):
    """
    Given a list of (name, func_source) for all functions in a seed group,
    build a func main() body that calls each non-method function with
    zero-value arguments.  Falls back to an empty body if signatures
    cannot be parsed.
    """
    calls = []
    for func_name, func_source in group_funcs:
        # Skip method functions (they need a receiver instance)
        if re.match(r'func\s*\(', func_source):
            continue

        # Locate the opening paren of the parameter list
        sig_m = re.match(r'func\s+\w+\s*\(', func_source)
        if not sig_m:
            continue
        paren_start = sig_m.end() - 1  # index of '('

        # Find matching ')'
        depth, params_end = 0, -1
        for j in range(paren_start, len(func_source)):
            if func_source[j] == '(':
                depth += 1
            elif func_source[j] == ')':
                depth -= 1
                if depth == 0:
                    params_end = j
                    break
        if params_end == -1:
            continue

        params_str = func_source[paren_start + 1:params_end]
        param_types = _parse_param_types(params_str)
        if param_types is None:
            continue

        zero_vals = [_zero_value_expr(t) for t in param_types]
        if any(z is None for z in zero_vals):
            continue

        calls.append(f'\t{func_name}({", ".join(zero_vals)})')

    if not calls:
        return 'func main() {}'
    return 'func main() {\n' + '\n'.join(calls) + '\n}'


# ---------------------------------------------------------------------------

def _is_runnable_seed(content: str) -> bool:
    """Return False for seeds that are known to fail under 'go run'."""
    if _RELATIVE_IMPORT_RE.search(content):
        return False
    if _INTERNAL_PKG_RE.search(content) or _INTERNAL_CMD_RE.search(content):
        return False
    if re.match(r'\s*//\s*errorcheck', content):
        return False
    pkg_match = re.search(r'^\s*package\s+(\w+)', content, re.MULTILINE)
    if pkg_match and pkg_match.group(1) not in ('main', 'test'):
        if '/internal/' in content or 'cmd/' in content:
            return False
    return True


def _make_shadow(content):
    """
    Return a copy of content where string literals and comments are replaced
    by space characters (preserving newlines and length), so we can safely
    count braces and search for keywords without false matches inside strings.
    """
    buf = list(content)
    i, n = 0, len(content)
    while i < n:
        c = content[i]
        # Line comment
        if c == '/' and i+1 < n and content[i+1] == '/':
            j = i
            while j < n and content[j] != '\n':
                buf[j] = ' '
                j += 1
            i = j
        # Block comment
        elif c == '/' and i+1 < n and content[i+1] == '*':
            j = i
            end = content.find('*/', i+2)
            end = (end + 2) if end != -1 else n
            while j < end:
                if content[j] != '\n':
                    buf[j] = ' '
                j += 1
            i = end
        # Raw string (backtick)
        elif c == '`':
            buf[i] = ' '
            j = i + 1
            while j < n and content[j] != '`':
                if content[j] != '\n':
                    buf[j] = ' '
                j += 1
            if j < n:
                buf[j] = ' '
                j += 1
            i = j
        # Interpreted string
        elif c == '"':
            buf[i] = ' '
            j = i + 1
            while j < n:
                if content[j] == '\\' and j+1 < n:
                    buf[j] = buf[j+1] = ' '
                    j += 2
                elif content[j] == '"':
                    buf[j] = ' '
                    j += 1
                    break
                else:
                    buf[j] = ' '
                    j += 1
            i = j
        # Rune literal
        elif c == "'":
            buf[i] = ' '
            j = i + 1
            while j < n and j < i + 6:
                if content[j] == '\\' and j+1 < n:
                    buf[j] = buf[j+1] = ' '
                    j += 2
                elif content[j] == "'":
                    buf[j] = ' '
                    j += 1
                    break
                else:
                    buf[j] = ' '
                    j += 1
            i = j
        else:
            i += 1
    return ''.join(buf)


def _find_top_level_funcs(shadow):
    """
    Find (start, end) byte positions of all top-level func declarations
    using the shadow string (comments/strings replaced by spaces).
    Positions point to the `func` keyword itself; callers may adjust them
    backward to absorb preceding //go: directives using the real content.
    """
    funcs = []
    n = len(shadow)

    for m in re.finditer(r'(?:^|\n)(func\s)', shadow):
        func_start = m.start(1)

        # Find the opening brace of the function body
        open_brace = shadow.find('{', func_start)
        if open_brace == -1:
            continue

        # Count braces to find end of function
        depth = 0
        j = open_brace
        while j < n:
            ch = shadow[j]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    funcs.append((func_start, j + 1))
                    break
            j += 1

    return funcs


def _extend_funcs_with_directives(func_positions, content):
    """
    For each (start, end) in func_positions, scan backward in the REAL
    content to absorb any contiguous //go: compiler directive lines that
    directly precede the func keyword.  Returns an updated list.

    This must operate on original content (not shadow) because //go: lines
    are comments and get erased in the shadow to spaces.
    """
    result = []
    for func_start, func_end in func_positions:
        scan_pos = func_start
        while True:
            line_end = scan_pos - 1
            if line_end < 0:
                break
            # Step back past the newline that ends the previous line
            if content[line_end] == '\n':
                line_end -= 1
            if line_end < 0:
                break
            line_start = content.rfind('\n', 0, line_end)
            line_start = line_start + 1 if line_start >= 0 else 0
            line_text = content[line_start:line_end + 1].strip()
            if line_text.startswith('//go:'):
                scan_pos = line_start
            else:
                break
        result.append((scan_pos, func_end))
    return result


def _extract_import_block(content):
    """
    Returns (import_block_string, content_without_imports).
    Handles both grouped `import (...)` and single `import "..."` forms.
    """
    # Grouped import
    m = re.search(r'\bimport\s*\(([^)]*)\)', content, re.DOTALL)
    if m:
        return m.group(0), content[:m.start()] + content[m.end():]

    # Single imports — collect all, remove from content, rebuild as grouped
    single_imports = re.findall(r'\bimport\s+"([^"]+)"', content)
    if single_imports:
        stripped = re.sub(r'\bimport\s+"[^"]+"\s*\n?', '', content)
        block = "import (\n" + "".join(f'\t"{p}"\n' for p in single_imports) + ")"
        return block, stripped

    return "", content


def _filter_imports_for_code(import_block, code):
    """
    Return an import block string containing only the imports referenced in code.
    Blank imports (_) and dot imports (.) are always kept.
    """
    if not import_block:
        return ""

    # Parse import specs: optional alias + path
    specs = re.findall(r'(\w+\s+)?"([^"]+)"', import_block)

    needed = []
    for alias_ws, path in specs:
        alias = alias_ws.strip() if alias_ws else ""
        if alias in ('_', '.'):
            needed.append((alias, path))
            continue
        pkg_name = alias if alias else path.split('/')[-1]
        if re.search(r'\b' + re.escape(pkg_name) + r'\b', code):
            needed.append((alias, path))

    if not needed:
        return ""
    if len(needed) == 1 and not needed[0][0]:
        return f'import "{needed[0][1]}"'

    lines = ["import ("]
    for alias, path in needed:
        if alias:
            lines.append(f'\t{alias} "{path}"')
        else:
            lines.append(f'\t"{path}"')
    lines.append(")")
    return "\n".join(lines)


def _split_go_file_into_seeds(content):
    """
    Split a Go file with multiple top-level functions into individual
    standalone package main programs, one per function.

    Returns a list of (func_name, standalone_code) tuples.
    Returns an empty list if splitting is not needed or appropriate.
    """
    shadow = _make_shadow(content)
    func_positions = _find_top_level_funcs(shadow)
    # Extend each function's start backward to absorb its //go: directives.
    # Must use original content (shadow has comments erased to spaces).
    func_positions = _extend_funcs_with_directives(func_positions, content)

    if len(func_positions) <= 1:
        return []

    # Skip small files — splitting them tends to produce more syntax errors
    # than useful seeds (e.g. helper files with just one or two tiny functions).
    if content.count('\n') + 1 < 100:
        return []

    # Header = everything before first function (package + imports + early decls)
    header_raw = content[:func_positions[0][0]].rstrip()

    # Normalise package to main
    header_raw = re.sub(r'\bpackage\s+\w+', 'package main', header_raw, count=1)

    # Separate import block from the rest of the header
    import_block, preamble = _extract_import_block(header_raw)
    preamble = preamble.strip()

    # Collect non-function declarations between functions AND after the last
    # function (type/var/const blocks).  Content after the last function is
    # frequently where helper variables like `var lt_0_uint64 = ...` live.
    between_decls = []
    for i in range(len(func_positions) - 1):
        _, end_i = func_positions[i]
        start_next, _ = func_positions[i+1]
        between = content[end_i:start_next].strip()
        if between:
            between_decls.append(between)
    # Also capture everything after the last function
    tail = content[func_positions[-1][1]:].strip()
    if tail:
        between_decls.append(tail)
    shared_decls = "\n\n".join(between_decls)

    # ------------------------------------------------------------------
    # Build per-function metadata and call-dependency groups
    # ------------------------------------------------------------------
    func_infos = []  # (name, start, end, source) for each kept function
    for start, end in func_positions:
        func_source = content[start:end]

        # Use search (not match) because func_source may start with //go: directives
        name_m = re.search(r'func\s+(?:\([^)]*\)\s+)?(\w+)', func_source)
        if not name_m:
            continue
        func_name = name_m.group(1)

        # Skip standard testing/benchmark/fuzz entry points and init
        if func_name == 'init' or re.match(r'(Test|Benchmark|Fuzz)\w*', func_name):
            continue

        # Skip functions that require the testing framework as a parameter
        sig_end = func_source.find('{')
        if sig_end != -1:
            sig = func_source[:sig_end]
            if any(t in sig for t in ('testing.T', 'testing.B', 'testing.M', 'testing.F')):
                continue

        func_infos.append((func_name, start, end, func_source))

    if not func_infos:
        return []

    # ------------------------------------------------------------------
    # Union-Find: group functions that call each other
    # We search the *original* source (not shadow) so that identifiers
    # referenced in comments/assembly annotations also trigger grouping —
    # it is safer to over-group than to split a function away from a
    # helper it actually needs.
    # ------------------------------------------------------------------
    parent = list(range(len(func_infos)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x, y):
        parent[_find(x)] = _find(y)

    all_names = [info[0] for info in func_infos]
    for i, (name_i, _, _, body_i) in enumerate(func_infos):
        for j, (name_j, _, _, _) in enumerate(func_infos):
            if i == j:
                continue
            # If func i's source mentions func j's name, they belong together
            if re.search(r'\b' + re.escape(name_j) + r'\b', body_i):
                _union(i, j)

    # Collect groups (ordered by first occurrence)
    from collections import defaultdict
    group_map = defaultdict(list)
    for i in range(len(func_infos)):
        group_map[_find(i)].append(i)

    # ------------------------------------------------------------------
    # Emit one seed per group
    # ------------------------------------------------------------------
    seeds = []
    group_name_seen = {}

    for root, indices in group_map.items():
        group = [func_infos[i] for i in indices]
        group_names = [info[0] for info in group]
        has_main = 'main' in group_names

        # Representative name: first function in source order
        group.sort(key=lambda x: x[1])  # sort by start position
        rep_name = group[0][0]

        # Deduplicate seed names
        count = group_name_seen.get(rep_name, 0)
        group_name_seen[rep_name] = count + 1
        seed_name = rep_name if count == 0 else f"{rep_name}_{count}"

        # Combined source for all functions in this group
        combined_funcs = "\n\n".join(info[3] for info in group)
        all_code = combined_funcs + "\n" + shared_decls

        # Filter imports to those referenced by this group
        filtered_imports = _filter_imports_for_code(import_block, all_code)

        # Assemble standalone program
        parts = []
        if preamble:
            parts.append(preamble)
        if filtered_imports:
            parts.append(filtered_imports)
        if shared_decls:
            parts.append(shared_decls)
        parts.append(combined_funcs)
        if not has_main:
            # Build a main() that actually calls the group's functions with
            # zero-value arguments so the seed exercises real code paths.
            main_body = _build_main_body([(info[0], info[3]) for info in group])
            parts.append(main_body)

        standalone = "\n\n".join(p.strip() for p in parts if p.strip())
        seeds.append((seed_name, standalone))

    return seeds


def parse_go_content(content):
    """
    Extracts basic Go metadata: imports, functions, types (structs/interfaces).
    """
    imports = re.findall(r'^\s*"([^"]+)"', content, re.MULTILINE)
    functions = re.findall(r'\bfunc\s+(?:\([^)]*\)\s+)?([a-zA-Z_]\w*)\s*\(', content)
    types = re.findall(r'\btype\s+([a-zA-Z_]\w*)\s+(?:struct|interface)\b', content)
    return {"imports": imports, "functions": functions, "types": types}


def collect_seeds(source_path, blacklist=None):
    if not os.path.exists(source_path):
        print(f"Error: Seed source path not found: {source_path}")
        return None

    seeds_output = []
    print(f"Scanning for .go files in {source_path}...")

    seed_paths = []
    try:
        for root, _, files in os.walk(source_path):
            for file in files:
                if file.endswith('.go'):
                    seed_paths.append(os.path.join(root, file))
    except OSError as e:
        print(f"Error listing directory: {e}")
        return None

    split_count = 0
    for file_path in seed_paths:
        seed_filename = os.path.relpath(file_path, source_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if not _is_runnable_seed(content):
                continue

            # Try to split multi-function files into individual seeds
            split_seeds = _split_go_file_into_seeds(content)

            if split_seeds:
                split_count += len(split_seeds)
                for func_name, standalone in split_seeds:
                    identifier = f"{seed_filename}::{func_name}"
                    metadata = parse_go_content(standalone)
                    metadata["type"] = "go"
                    metadata["filename"] = identifier
                    metadata["split_from"] = seed_filename
                    seeds_output.append({
                        "identifier": identifier,
                        "content": standalone,
                        "metadata": metadata,
                    })
            else:
                # Single-function file or unsplittable: use as-is
                metadata = parse_go_content(content)
                metadata["type"] = "go"
                metadata["filename"] = seed_filename
                seeds_output.append({
                    "identifier": seed_filename,
                    "content": content,
                    "metadata": metadata,
                })

        except Exception:
            pass

    print(f"Collected {len(seeds_output)} seeds "
          f"({split_count} from function-level splitting).")

    # Save to DB
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "corpus.db")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS seeds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT UNIQUE,
                content TEXT,
                metadata TEXT
            )
        ''')

        count = 0
        for seed in seeds_output:
            try:
                cursor.execute(
                    "INSERT INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)",
                    (seed['identifier'], seed['content'], json.dumps(seed['metadata']))
                )
                count += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()
        print(f"Saved {count} seeds to {db_path}")
        return db_path
    except Exception as e:
        print(f"Error saving to corpus: {e}")
        return None


def load_corpus(db_path):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT identifier, content, metadata FROM seeds")
    rows = cursor.fetchall()
    conn.close()

    return [
        {"filename": r[0], "content": r[1], "metadata": json.loads(r[2])}
        for r in rows
    ]
