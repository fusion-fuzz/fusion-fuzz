"""
projects/tvm/parser.py — TVMScript seed collection for FusionFuzz.

Seeds are Python files each defining one `@I.ir_module class Module`
(produced from TVM's own tests by projects/tvm/extract_seeds.py). The
metadata records the function kinds inside the module and a coarse
identifier-co-occurrence dataflow, as the other Python-shaped adapters do.
"""

import os
import re

from core.parser import BaseParser

_KEYWORDS = frozenset("""
def class return if elif else for while in not and or is pass break continue
raise try except finally with as import from True False None lambda yield
""".split())
_IDENT_RE = re.compile(r'\b[A-Za-z_]\w*\b')
_STR_RE = re.compile(r'"""[\s\S]*?"""|"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')


class TVMParser(BaseParser):
    extensions = ['.py']
    seed_type = 'tvm'

    def parse_content(self, content, filename=""):
        code = _STR_RE.sub(' ', content)
        groups, allv = [], []
        for line in code.splitlines():
            line = line.split("#", 1)[0]
            idents = [t for t in _IDENT_RE.findall(line) if t not in _KEYWORDS and not t[0].isdigit()]
            idents = list(dict.fromkeys(idents))
            if idents:
                allv.extend(idents); groups.append(idents)
        variables = list(dict.fromkeys(allv))
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for g in groups:
            for v in g[1:]:
                parent[find(v)] = find(g[0])
        flows = {}
        for v in variables:
            flows.setdefault(find(v), []).append(v)
        return {
            "type": "tvm",
            "extension": ".py",
            "has_prim": "@T.prim_func" in content,
            "has_relax": "@R.function" in content,
            "funcs": re.findall(r'^\s{4}def\s+([A-Za-z_]\w*)', content, re.M),
            "gpu": bool(re.search(r'T\.launch_thread|threadIdx|blockIdx|"cuda"|"vulkan"|"metal"|"opencl"', content)),
            "variables": variables,
            "dataflows": [sorted(v) for v in flows.values() if len(v) > 1],
            "line_count": len(content.splitlines()),
        }


_parser = TVMParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
