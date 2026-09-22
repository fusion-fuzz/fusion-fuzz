"""
projects/mojo/parser.py — Mojo seed collection for FusionFuzz.

Seeds are .mojo files under projects/mojo/seeds/ (populated by setup.py
from Mojo/stdlib/test, Mojo/test/{mojo-parser,mojo-integration,mojo-tool}
and Mojo/examples of the modular/modular checkout). Metadata records what
the fusion strategies and the driver need: whether the file defines
`main`, its module-level names, its imports, and a coarse dataflow signal
(identifiers co-occurring on a line, the same heuristic the C and Python
adapters use — Mojo is Python-shaped and `ast` cannot parse it).
"""

import os
import re

from core.parser import BaseParser

_KEYWORDS = frozenset("""
def struct trait alias var comptime return if elif else for while in not and or
is pass break continue raise raises try except finally with as import from
True False None self Self ref mut out owned read print len range
""".split())
_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
_STR_RE = re.compile(r'"""[\s\S]*?"""|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')
_TOP_RE = re.compile(r'^(def|struct|trait|alias|var)\s+([A-Za-z_]\w*)', re.M)
_IMPORT_RE = re.compile(r'^(?:from\s+(\S+)\s+import|import\s+(\S+))', re.M)


class MojoFastDataflow:
    def analyze(self, code):
        code = _STR_RE.sub(' ', code)
        line_groups, all_vars = [], []
        for line in code.splitlines():
            line = line.split("#", 1)[0]
            idents = [t for t in _IDENT_RE.findall(line) if t not in _KEYWORDS and not t[0].isdigit()]
            idents = list(dict.fromkeys(idents))
            if idents:
                all_vars.extend(idents)
                line_groups.append(idents)
        variables = list(dict.fromkeys(all_vars))
        # union-find over line groups
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for g in line_groups:
            for v in g[1:]:
                parent[find(v)] = find(g[0])
        flows = {}
        for v in variables:
            flows.setdefault(find(v), []).append(v)
        return variables, [sorted(v) for v in flows.values() if len(v) > 1]


class MojoParser(BaseParser):
    extensions = ['.mojo']
    seed_type = 'mojo'

    def parse_content(self, content, filename=""):
        variables, dataflows = MojoFastDataflow().analyze(content)
        return {
            "type": "mojo",
            "extension": ".mojo",
            "has_main": bool(re.search(r'^def\s+main\s*\(', content, re.M)),
            "top_level": [m.group(2) for m in _TOP_RE.finditer(content)],
            "imports": [m.group(1) or m.group(2) for m in _IMPORT_RE.finditer(content)],
            "is_test_suite": "TestSuite" in content,
            "expects_diagnostics": "expected-error" in content,
            "variables": variables,
            "dataflows": dataflows,
            "line_count": len(content.splitlines()),
        }


_parser = MojoParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
