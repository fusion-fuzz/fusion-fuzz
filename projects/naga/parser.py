import re
from core.parser import BaseParser


_WGSL_KEYWORDS = frozenset({
    "alias", "array", "atomic", "bitcast", "bool", "break", "case", "const",
    "const_assert", "continue", "continuing", "default", "diagnostic",
    "discard", "else", "enable", "false", "fn", "for", "function", "if",
    "let", "loop", "mat2x2", "mat2x3", "mat2x4", "mat3x2", "mat3x3",
    "mat3x4", "mat4x2", "mat4x3", "mat4x4", "override", "private",
    "ptr", "requires", "return", "select", "storage", "struct", "switch",
    "true", "type", "var", "vec2", "vec3", "vec4", "while", "workgroup",
    "read", "write", "read_write", "i32", "u32", "f32", "f16", "abstract",
})

_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
_LINE_COMMENT_RE = re.compile(r'//.*$')
_BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.S)
_STR_RE = re.compile(r'"(?:[^"\\]|\\.)*"')
_DECL_RE = re.compile(
    r'\b(?:var(?:\s*<[^>]+>)?|let|const|override)\s+([A-Za-z_]\w*)'
    r'\s*(?::\s*([^=;]+))?'
)


class WGSLFastDataflow:
    """Coarse WGSL def/use grouping by non-comment line co-occurrence."""

    def analyze(self, code: str):
        code = _BLOCK_COMMENT_RE.sub(" ", code)
        variables = []
        dataflows = []
        for line in code.splitlines():
            line = _LINE_COMMENT_RE.sub("", line)
            line = _STR_RE.sub(" ", line)
            idents = [t for t in _IDENT_RE.findall(line) if t not in _WGSL_KEYWORDS]
            idents = list(dict.fromkeys(idents))
            if idents:
                variables.extend(idents)
                dataflows.append(idents)
        return list(dict.fromkeys(variables)), self._merge(dataflows)

    @staticmethod
    def _merge(groups):
        merged = []
        for group in groups:
            target = next((m for m in merged if any(v in m for v in group)), None)
            if target is None:
                merged.append(list(group))
            else:
                for v in group:
                    if v not in target:
                        target.append(v)
        return merged


class NagaParser(BaseParser):
    extensions = [".wgsl"]
    seed_type = "wgsl"

    _FN_RE = re.compile(r'\bfn\s+([A-Za-z_]\w*)\s*\(')
    _STRUCT_RE = re.compile(r'\bstruct\s+([A-Za-z_]\w*)\s*\{')
    _ALIAS_RE = re.compile(r'\balias\s+([A-Za-z_]\w*)\s*=')

    def parse_content(self, content, filename=""):
        variables, dataflows = WGSLFastDataflow().analyze(content)
        declarations = (
            self._STRUCT_RE.findall(content)
            + self._ALIAS_RE.findall(content)
            + self._FN_RE.findall(content)
        )
        var_types = {}
        for m in _DECL_RE.finditer(content):
            name, ty = m.group(1), m.group(2)
            if ty:
                var_types[name] = re.sub(r'\s+', ' ', ty.strip())
        return {
            "type": "wgsl",
            "extension": ".wgsl",
            "variables": variables,
            "dataflows": dataflows,
            "functions": self._FN_RE.findall(content),
            "structs": self._STRUCT_RE.findall(content),
            "aliases": self._ALIAS_RE.findall(content),
            "declarations": list(dict.fromkeys(declarations)),
            "var_types": var_types,
            "has_declaration": bool(declarations),
        }


_parser = NagaParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
