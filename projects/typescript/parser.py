"""
projects/typescript/parser.py — turn .ts files into seeds.

Variables are the file's declared names (values and types); dataflow
groups are line co-occurrence, the same approximation the C-family
parser uses. The fusion strategies rescan the text themselves.
"""

import os
import re

from core.parser import BaseParser

_DECL_RE = re.compile(
    r'\b(?:let|const|var|function\*?|class|interface|type|enum|namespace|module)\s+([A-Za-z_$][\w$]*)')
_KEYWORDS = frozenset("""
let const var function class interface type enum namespace module import export
default return if else for while do switch case break continue new this super
typeof instanceof in of void delete throw try catch finally yield await async
as is keyof readonly infer declare abstract implements extends static public
private protected get set constructor any unknown never number string boolean
symbol object bigint undefined null true false
""".split())


class TSFastDataflow:
    def analyze(self, code):
        variables = set()
        groups = []
        for line in code.splitlines():
            line = line.split("//", 1)[0]
            names = {n for n in _DECL_RE.findall(line) if n not in _KEYWORDS}
            variables |= names
            used = {n for n in re.findall(r'(?<![\w$.])([A-Za-z_$][\w$]*)', line) if n in variables}
            names |= used
            if len(names) > 1:
                groups.append(sorted(names))
        merged = []
        for g in groups:
            gs = set(g)
            keep = []
            for m in merged:
                if m & gs:
                    gs |= m
                else:
                    keep.append(m)
            keep.append(gs)
            merged = keep
        return sorted(variables), [sorted(m) for m in merged if len(m) > 1]


class TypeScriptParser(BaseParser):
    extensions = ['.ts']
    seed_type = 'typescript'

    def parse_content(self, content, filename=""):
        variables, dataflows = TSFastDataflow().analyze(content)
        return {
            "type": "typescript",
            "extension": ".ts",
            "is_test": True,
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = TypeScriptParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
