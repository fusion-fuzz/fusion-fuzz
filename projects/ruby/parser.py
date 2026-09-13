"""
projects/ruby/parser.py — turn .rb files into seeds.

Variables are the names a script assigns (locals, instance/global
variables, constants); dataflow groups are line co-occurrence, the same
approximation the C-family parser uses. Ruby has no cheap AST from
Python, and the fusion strategies in core/fusion.py rescan the seed text
themselves, so this only has to be good enough for --pre-analysis's
co-occurrence groups.
"""

import os
import re

from core.parser import BaseParser

_ASSIGN_RE = re.compile(
    r'(?<![\w.:@$])((?:@@|@|\$)?[A-Za-z_]\w*)\s*(?:\|\||&&|\*\*|<<|>>|[-+*/%|&^])?=(?![=~>])')
_BLOCK_PARAM_RE = re.compile(r'\|\s*([^|]*?)\s*\|')
_KEYWORDS = frozenset("""
BEGIN END alias and begin break case class def defined? do else elsif end
ensure false for if in module next nil not or redo rescue retry return self
super then true undef unless until when while yield __method__ __FILE__
__LINE__ __dir__ lambda proc loop puts print p pp require require_relative
raise attr_accessor attr_reader attr_writer include extend prepend private
public protected module_function new
""".split())


class RubyFastDataflow:
    def analyze(self, code):
        variables = set()
        groups = []
        for line in code.splitlines():
            line = line.split("#", 1)[0]
            names = set()
            for m in _ASSIGN_RE.finditer(line):
                n = m.group(1)
                if n.lstrip("@$") not in _KEYWORDS:
                    names.add(n)
                    variables.add(n)
            for m in _BLOCK_PARAM_RE.finditer(line):
                for n in re.findall(r'[A-Za-z_]\w*', m.group(1)):
                    if n not in _KEYWORDS:
                        variables.add(n)
                        names.add(n)
            used = {n for n in re.findall(r'(?<![\w.:@$])([a-z_]\w*)(?![\w?!(])', line)
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


class RubyParser(BaseParser):
    extensions = ['.rb']
    seed_type = 'ruby'

    def parse_content(self, content, filename=""):
        variables, dataflows = RubyFastDataflow().analyze(content)
        return {
            "type": "ruby",
            "extension": ".rb",
            "is_test": (filename or "").startswith("test_"),
            "variables": variables,
            "dataflows": dataflows,
        }


_parser = RubyParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    collect_seeds(os.path.join(script_dir, "seeds"))
