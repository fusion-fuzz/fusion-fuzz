"""
projects/julia/parser.py — turn .jl files into seeds.

Variables are the names a script assigns or defines (locals, functions,
types); dataflow groups are line co-occurrence, the same approximation
the C-family parser uses. The fusion strategies rescan the seed text
themselves.
"""

import os
import re

from core.parser import BaseParser

_ASSIGN_RE = re.compile(r'(?<![\w.])([A-Za-z_][\w!]*)\s*(?::\:[^=]*)?=(?![=>])')
_FUNC_RE = re.compile(r'^\s*function\s+([A-Za-z_][\w!]*)|^\s*([A-Za-z_][\w!]*)\s*\([^)]*\)\s*=(?![=>])', re.M)
_TYPE_RE = re.compile(r'^\s*(?:mutable\s+)?struct\s+([A-Za-z_]\w*)|^\s*abstract\s+type\s+([A-Za-z_]\w*)', re.M)
_KEYWORDS = frozenset("""
function end if else elseif for while begin let do try catch finally return
break continue struct mutable abstract type primitive module using import
export const global local quote macro where in isa true false nothing missing
Inf NaN print println push! length size zeros ones rand collect map filter
reduce sum test Test
""".split())


class JuliaFastDataflow:
    def analyze(self, code):
        variables, groups = set(), []
        for line in code.splitlines():
            line = line.split("#", 1)[0]
            names = {n for n in _ASSIGN_RE.findall(line) if n not in _KEYWORDS}
            variables |= names
            used = {n for n in re.findall(r'(?<![\w.@])([A-Za-z_][\w!]*)', line)
                    if n in variables}
            names |= used
            if len(names) > 1:
                groups.append(sorted(names))
        for rx in (_FUNC_RE, _TYPE_RE):
            for m in rx.finditer(code):
                for g in m.groups():
                    if g and g not in _KEYWORDS:
                        variables.add(g)
        merged = []
        for g in groups:
            gs = set(g)
            keep = []
            for mset in merged:
                if mset & gs:
                    gs |= mset
                else:
                    keep.append(mset)
            keep.append(gs)
            merged = keep
        return sorted(variables), [sorted(m) for m in merged if len(m) > 1]


class JuliaParser(BaseParser):
    extensions = ['.jl']
    seed_type = 'julia'

    def parse_content(self, content, filename=""):
        variables, dataflows = JuliaFastDataflow().analyze(content)
        return {
            "type": "julia",
            "extension": ".jl",
            "is_test": (filename or "").startswith("ts_"),
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = JuliaParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
