import os
import re
from core.parser import BaseParser

# ==========================================
# Coarse-grained C/C++ Dataflow Logic
# (same spirit as PHPFastDataflow in projects/php/parser.py, adapted for
#  unmarked C identifiers instead of $-prefixed PHP variables)
# ==========================================

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

_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
_LINE_COMMENT_RE = re.compile(r'//.*$')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)
_PP_LINE_RE = re.compile(r'^\s*#')
_STR_CHAR_RE = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'')


class CFastDataflow:
    """
    Fast, coarse-grained dataflow analysis on C/C++ source: identifiers that
    co-occur on the same (non-comment, non-preprocessor) line are grouped
    into the same dataflow, mirroring PHPFastDataflow's line-co-occurrence
    heuristic. Not complete, aims to be a useful (not perfect) signal for
    picking "related" bridge variables during dataflow fusion.
    """

    def analyze(self, code: str):
        code = _BLOCK_COMMENT_RE.sub(' ', code)
        line_groups = []
        all_vars = []
        for line in code.splitlines():
            if _PP_LINE_RE.match(line):
                continue
            line = _LINE_COMMENT_RE.sub('', line)
            line = _STR_CHAR_RE.sub(' ', line)
            idents = [t for t in _IDENT_RE.findall(line) if t not in _C_KEYWORDS]
            idents = list(dict.fromkeys(idents))
            if not idents:
                continue
            all_vars.extend(idents)
            line_groups.append(idents)

        variables = list(dict.fromkeys(all_vars))
        dataflows = self._merge_dataflows(line_groups)
        return variables, dataflows

    @staticmethod
    def _merge_dataflows(line_groups):
        merged = []
        for group in line_groups:
            joined = False
            for existing in merged:
                if any(v in existing for v in group):
                    for v in group:
                        if v not in existing:
                            existing.append(v)
                    joined = True
                    break
            if not joined:
                merged.append(list(group))
        return merged


# --c / --cpp / --m (main.py) select which of clang's input languages this
# project fuzzes. Each language owns its own source extensions; .mm is
# Objective-C++, so it only belongs in the corpus when *both* C++ and
# Objective-C are enabled — a .mm seed is not compilable as either one
# alone.
LANG_EXTENSIONS = {
    'c':    ['.c'],
    'cpp':  ['.cpp', '.cc', '.cxx'],
    'objc': ['.m'],
}
ALL_EXTENSIONS = ['.c', '.cpp', '.cc', '.cxx', '.m', '.mm']


def allowed_extensions(langs=None):
    """Source extensions for a set of language keys ('c'/'cpp'/'objc').
    An empty/None set means "no restriction" — every extension, which is
    the behavior from before the flags existed."""
    if not langs:
        return list(ALL_EXTENSIONS)
    exts = []
    for lang in ('c', 'cpp', 'objc'):
        if lang in langs:
            exts.extend(LANG_EXTENSIONS[lang])
    if 'cpp' in langs and 'objc' in langs:
        exts.append('.mm')
    return exts


class ClangParser(BaseParser):
    extensions = list(ALL_EXTENSIONS)
    seed_type = 'c'

    _CXX_EXTS = ('.cpp', '.cc', '.cxx', '.mm')

    def set_languages(self, langs=None):
        """Restrict both seed collection (which files are parsed) and
        corpus loading (which stored seeds are used) to `langs`."""
        self.extensions = allowed_extensions(langs)

    def load_corpus(self, db_path: str) -> list:
        seeds = super().load_corpus(db_path)
        if set(self.extensions) == set(ALL_EXTENSIONS):
            return seeds
        # A corpus.db built by an earlier (or unrestricted) run holds every
        # language, so filter on load too rather than assuming the DB was
        # collected with the same flags.
        allowed = set(self.extensions)
        return [s for s in seeds
                if (s["metadata"].get("extension") or ".c").lower() in allowed]

    def parse_content(self, content, filename=""):
        ext = os.path.splitext(filename)[1].lower()
        if ext == '.mm':
            seed_type = 'objcpp'
        elif ext in self._CXX_EXTS:
            seed_type = 'cpp'
        elif ext == '.m':
            seed_type = 'objc'
        else:
            seed_type = 'c'

        variables, dataflows = CFastDataflow().analyze(content)

        return {
            "type": seed_type,
            "extension": ext,
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = ClangParser(__file__)


def set_languages(langs=None):
    _parser.set_languages(langs)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "llvm-project", "clang", "test")
    print("Executing Clang parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: seed source not found at {seeds_dir}. Run setup.py first.")
