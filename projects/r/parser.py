"""
projects/r/parser.py — turn .R files into seeds.

Variables are the names a script assigns (`x <- 1`, `f = function(...)`,
`x <<- 1`); dataflow groups are line co-occurrence, the same
approximation the C-family parser uses. The fusion strategies rescan the
seed text themselves, so this only has to be good enough for
--pre-analysis's co-occurrence groups.
"""

import os
import re

from core.parser import BaseParser

_ASSIGN_RE = re.compile(r'(?<![\w.$@])([A-Za-z._][\w._]*)\s*(?:<<?-|=(?!=))')
_ARROW_RIGHT_RE = re.compile(r'->>?\s*([A-Za-z._][\w._]*)')
_KEYWORDS = frozenset("""
if else for while repeat function return break next TRUE FALSE NULL NA Inf
NaN in library require c list vector matrix data frame print cat paste
paste0 length names dim nrow ncol sum mean seq rep which sapply lapply
vapply mapply apply Reduce Filter Map do.call stopifnot invisible
""".split())


class RFastDataflow:
    def analyze(self, code):
        variables, groups = set(), []
        for line in code.splitlines():
            line = line.split("#", 1)[0]
            names = {n for n in _ASSIGN_RE.findall(line) if n not in _KEYWORDS}
            names |= {n for n in _ARROW_RIGHT_RE.findall(line) if n not in _KEYWORDS}
            variables |= names
            used = {n for n in re.findall(r'(?<![\w.$@])([A-Za-z._][\w._]*)', line)
                    if n in variables}
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


class RParser(BaseParser):
    extensions = ['.R']
    seed_type = 'r'

    def parse_content(self, content, filename=""):
        variables, dataflows = RFastDataflow().analyze(content)
        return {
            "type": "r",
            "extension": ".R",
            "is_test": (filename or "").startswith(("tests_", "lib_")),
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = RParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    collect_seeds(os.path.join(os.path.dirname(os.path.abspath(__file__)), "seeds"))
