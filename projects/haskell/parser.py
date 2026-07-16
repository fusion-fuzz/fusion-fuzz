import re

from core.parser import BaseParser

# ==========================================
# Shadowing — blank out strings/chars/comments so regexes over the real
# source never misfire inside string/char literals or comments. Haskell
# block comments ({- -}) nest; we approximate with non-nested handling,
# which is fine for the vast majority of seeds.
# ==========================================

def _make_shadow(content: str) -> str:
    buf = list(content)
    i, n = 0, len(content)
    while i < n:
        c = content[i]
        # Line comment
        if c == '-' and i + 1 < n and content[i + 1] == '-' and (
            i + 2 >= n or content[i + 2] not in '!#$%&*+./<=>?@\\^|~-'
        ):
            j = i
            while j < n and content[j] != '\n':
                buf[j] = ' '
                j += 1
            i = j
        # Block comment {- ... -} (non-nested approximation)
        elif c == '{' and i + 1 < n and content[i + 1] == '-':
            j = i
            end = content.find('-}', i + 2)
            end = (end + 2) if end != -1 else n
            while j < end:
                if content[j] != '\n':
                    buf[j] = ' '
                j += 1
            i = end
        # String literal
        elif c == '"':
            buf[i] = ' '
            j = i + 1
            while j < n:
                if content[j] == '\\' and j + 1 < n:
                    buf[j] = buf[j + 1] = ' '
                    j += 2
                elif content[j] == '"':
                    buf[j] = ' '
                    j += 1
                    break
                else:
                    if content[j] != '\n':
                        buf[j] = ' '
                    j += 1
            i = j
        # Char literal — bounded lookahead so we don't eat a lone "'" used
        # as an identifier suffix (e.g. x', map').
        elif c == "'" and i > 0 and not re.match(r"[A-Za-z0-9_']", content[i - 1]):
            j = i + 1
            closed = False
            if j < n and content[j] == '\\':
                k = j + 1
                while k < n and content[k] != "'" and k < j + 6:
                    k += 1
                if k < n and content[k] == "'":
                    closed = True
                    j = k
            elif j + 1 < n and content[j + 1] == "'":
                closed = True
                j = j + 1
            if closed:
                for k in range(i, j + 1):
                    if content[k] != '\n':
                        buf[k] = ' '
                i = j + 1
            else:
                i += 1
        else:
            i += 1
    return ''.join(buf)


# ==========================================
# Regex-based metadata extraction
# ==========================================

_PRAGMA_RE = re.compile(r'\{-#.*?#-\}', re.DOTALL)
_MODULE_RE = re.compile(r'^\s*module\s+([A-Za-z0-9_.\']+)')

_HS_KEYWORDS = frozenset({
    'data', 'newtype', 'type', 'class', 'instance', 'where', 'let', 'in',
    'do', 'if', 'then', 'else', 'case', 'of', 'import', 'module', 'deriving',
    'infixl', 'infixr', 'infix', 'foreign', 'default', 'family', 'forall',
    'main',
})

_TOPLEVEL_FUNC_RE = re.compile(r"^([a-z_][A-Za-z0-9_']*)\s*(?:::|[^=\n]*=)", re.MULTILINE)
_TOPLEVEL_TYPE_RE = re.compile(r"^(?:data|newtype|type)\s+([A-Z][A-Za-z0-9_']*)", re.MULTILINE)
_TOPLEVEL_CLASS_RE = re.compile(r"^class\s+(?:.*=>\s*)?([A-Z][A-Za-z0-9_']*)", re.MULTILINE)

_NULLARY_RE = re.compile(r"^([a-z_][A-Za-z0-9_']*)\s*=(?!=)", re.MULTILINE)
_MAIN_RE = re.compile(r"^main\s*(?:::|=)", re.MULTILINE)

# Handle-creation sites: `name <- newIORef ...`, `name <- newMVar ...`, etc.
_STATE_NEW_RE = re.compile(
    r"\b([a-z_][A-Za-z0-9_']*)\s*<-\s*"
    r"(newIORef|newMVar|newEmptyMVar|newTVarIO|atomically\s*\(\s*newTVar)\b"
)

_STATE_USE_FUNCS = (
    'readIORef', 'writeIORef', 'modifyIORef', "modifyIORef'", 'atomicModifyIORef',
    "atomicModifyIORef'", 'takeMVar', 'putMVar', 'readMVar', 'modifyMVar',
    'modifyMVar_', 'swapMVar', 'readTVarIO', 'readTVar', 'writeTVar', 'modifyTVar',
    "modifyTVar'",
)
_STATE_USE_RE = re.compile(
    r"\b(?:%s)\s+([a-z_][A-Za-z0-9_']*)" % '|'.join(_STATE_USE_FUNCS)
)

def _classify_ctor(ctor: str) -> str:
    if 'TVar' in ctor:
        return 'tvar'
    if 'MVar' in ctor:
        return 'mvar'
    return 'ioref'


class HaskellParser(BaseParser):
    extensions = ['.hs']
    seed_type = 'haskell'

    def parse_content(self, content: str, filename: str = "") -> dict:
        shadow = _make_shadow(content)

        pragmas = [m.strip() for m in _PRAGMA_RE.findall(content)]
        # Match against shadow (to anchor line boundaries safely) but keep
        # the real text — import lines essentially never contain strings.
        imports = [ln.strip() for ln in content.splitlines()
                   if re.match(r'^\s*import\s+', ln)]

        mod_match = _MODULE_RE.search(shadow)
        module_name = mod_match.group(1) if mod_match else None

        toplevel = set()
        toplevel |= {m.group(1) for m in _TOPLEVEL_FUNC_RE.finditer(shadow)}
        toplevel |= {m.group(1) for m in _TOPLEVEL_TYPE_RE.finditer(shadow)}
        toplevel |= {m.group(1) for m in _TOPLEVEL_CLASS_RE.finditer(shadow)}
        toplevel -= _HS_KEYWORDS

        nullary = sorted({m.group(1) for m in _NULLARY_RE.finditer(shadow)} - _HS_KEYWORDS)

        has_main = bool(_MAIN_RE.search(shadow))

        state_handles = []
        seen_handles = set()
        for m in _STATE_NEW_RE.finditer(shadow):
            name, ctor = m.group(1), m.group(2)
            kind = _classify_ctor(ctor)
            if name not in seen_handles:
                seen_handles.add(name)
                state_handles.append({"name": name, "kind": kind})

        state_used = sorted({m.group(1) for m in _STATE_USE_RE.finditer(shadow)})

        return {
            "pragmas": sorted(set(pragmas)),
            "imports": sorted(set(imports)),
            "module_name": module_name,
            "toplevel_names": sorted(toplevel),
            "nullary_bindings": nullary,
            "has_main": has_main,
            "state_handles": state_handles,
            "state_used": state_used,
        }


_parser = HaskellParser(__file__)


def collect_seeds(source_path: str, blacklist: list = None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path: str):
    return _parser.load_corpus(db_path)
