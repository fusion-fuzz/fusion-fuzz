"""
projects/tint/parser.py — turn WGSL files into seeds for tint.

tint and naga are both WGSL compilers, so the parsing is identical: the
same WGSLFastDataflow def/use grouping and the same declaration/struct/
alias extraction the fusion strategies consume. This reuses naga's parser
wholesale rather than copying it — the arrangement GCC has with clang's
parser and SpiderMonkey has with V8's. Only the driver and the crash
oracle are tint-specific.
"""

import importlib.util
import os

from core.parser import BaseParser

# main.py and core/parser.py load project parsers by file path, so there is
# no package context for a relative import. The absolute form works because
# FFL runs from the repo root.
try:
    from projects.naga.parser import WGSLFastDataflow
except ImportError:  # pragma: no cover - direct-load fallback
    _naga = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "naga", "parser.py")
    _spec = importlib.util.spec_from_file_location("ffl_naga_parser", _naga)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    WGSLFastDataflow = _mod.WGSLFastDataflow

import re


class TintParser(BaseParser):
    extensions = [".wgsl"]
    seed_type = "wgsl"

    _FN_RE = re.compile(r'\bfn\s+([A-Za-z_]\w*)\s*\(')
    _STRUCT_RE = re.compile(r'\bstruct\s+([A-Za-z_]\w*)\s*\{')
    _ALIAS_RE = re.compile(r'\balias\s+([A-Za-z_]\w*)\s*=')
    _DECL_RE = re.compile(
        r'\b(?:var(?:\s*<[^>]+>)?|let|const|override)\s+([A-Za-z_]\w*)'
        r'\s*(?::\s*([^=;]+))?')

    def parse_content(self, content, filename=""):
        variables, dataflows = WGSLFastDataflow().analyze(content)
        declarations = (self._STRUCT_RE.findall(content)
                        + self._ALIAS_RE.findall(content)
                        + self._FN_RE.findall(content))
        var_types = {}
        for m in self._DECL_RE.finditer(content):
            name, ty = m.group(1), m.group(2)
            if ty:
                var_types[name] = re.sub(r'\s+', ' ', ty.strip())
        return {
            # core/dryrun.py's _COLLECTORS keys on this; "wgsl" reuses the
            # WGSL metadata collector naga registered.
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


_parser = TintParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing tint parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
