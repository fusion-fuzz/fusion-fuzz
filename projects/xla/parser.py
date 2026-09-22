"""
projects/xla/parser.py — turn XLA's HLO test modules into seeds.

Two seed sources, both under projects/xla/xla/xla:

  *.hlo / *.hlotxt   ~500 standalone HLO text modules (lit tests, benchmark
                     inputs, extracted repros).
  *_test.cc          ~9,000 HLO modules embedded as C++ raw strings —
                     `R"(HloModule ...)"` / `R"hlo(...)hlo"` — one per test
                     case of XLA's pass, verifier and backend tests. This is
                     where XLA's test suite actually lives; the standalone
                     files are the minority. Every raw string that contains
                     an `HloModule` header and at least one computation
                     becomes a seed named `<file>#<n>`.

Per-seed metadata is the HLO analysis in projects/xla/analyzer.py
(computations, the entry's parameters and their shapes, opcodes), which the
fusion strategies in core/fusion.py consume to merge two modules by shape.
`variables`/`dataflows` are the entry instructions and their def-use edges
so the generic dataflow machinery sees a name pool.
"""

import importlib.util
import os
import re

from core.parser import BaseParser

try:
    from projects.xla.analyzer import analyze_seed
except ImportError:  # pragma: no cover - direct-load fallback
    _spec = importlib.util.spec_from_file_location(
        "ffl_xla_analyzer_parser",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    analyze_seed = _mod.analyze_seed


# C++11 raw string literal with an optional delimiter: R"(...)", R"hlo(...)hlo".
_RAW_STRING_RE = re.compile(r'R"(\w*)\((.*?)\)\1"', re.S)

# Lines that lit tests put in front of the module and the HLO parser does
# not read (`// RUN:` is a comment to it, so it would parse — but a fused
# child inherits two RUN lines, and hlo-opt is not invoked through lit
# here anyway).
_RUN_LINE_RE = re.compile(r'^\s*//\s*(?:RUN|CHECK[^:]*|REQUIRES|XFAIL|UNSUPPORTED):.*$', re.M)


def _strip_lit(text):
    return _RUN_LINE_RE.sub("", text)


def _looks_like_module(text, inline=False):
    """A text is an HLO module when it has a computation body; an inline C++
    raw string additionally needs the header or an ENTRY keyword, or every
    `{...}` literal in a test would be taken for one. Header-less files
    (`e { ... }`, 319 of the 517 .hlo files) are modules too."""
    if "{" not in text or "}" not in text:
        return False
    if inline:
        return "HloModule" in text or "ENTRY" in text
    return "HloModule" in text or _COMP_OPEN_RE.search(text) is not None


_COMP_OPEN_RE = re.compile(r'^\s*(?:ENTRY\s+)?%?[\w.\-]+\s*(?:\(.*?\)\s*->\s*.+?)?\s*\{\s*$', re.M)


class XLAParser(BaseParser):
    extensions = ['.hlo', '.hlotxt']
    seed_type = 'hlo'

    def parse_content(self, content, filename=""):
        meta = analyze_seed(content, filename)
        meta["extension"] = ".hlo"
        return meta

    # Beyond the .hlo files BaseParser finds by extension, walk the C++
    # tests for embedded modules.
    def collect_seeds(self, source_path, blacklist=None):
        if not os.path.exists(source_path):
            print(f"Error: Seed source path not found: {source_path}")
            return None
        seeds = []
        n_files = n_inline = 0
        for root, _, files in os.walk(source_path):
            for fname in files:
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, source_path)
                if any(fname.endswith(ext) for ext in self.extensions):
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            content = _strip_lit(f.read())
                    except OSError:
                        continue
                    if not _looks_like_module(content):
                        continue
                    if blacklist and any(t in content for t in blacklist):
                        continue
                    seeds.append(self._seed(rel, content))
                    n_files += 1
                elif fname.endswith("_test.cc"):
                    try:
                        with open(path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                    except OSError:
                        continue
                    if "HloModule" not in text:
                        continue
                    k = 0
                    for m in _RAW_STRING_RE.finditer(text):
                        body = m.group(2)
                        if not _looks_like_module(body, inline=True):
                            continue
                        # C++ tests often indent the whole literal; the HLO
                        # parser does not care, but dedent for readability.
                        body = _dedent(body).strip("\n") + "\n"
                        if blacklist and any(t in body for t in blacklist):
                            continue
                        seeds.append(self._seed(f"{rel}#{k}", body))
                        k += 1
                        n_inline += 1
        print(f"Found {n_files} HLO files and {n_inline} inline modules in C++ tests.")
        return self._save_to_db(seeds)

    def _seed(self, identifier, content):
        metadata = self.parse_content(content, identifier)
        metadata.setdefault("type", self.seed_type)
        metadata["filename"] = identifier
        return {"identifier": identifier, "content": content, "metadata": metadata}


def _dedent(text):
    lines = text.splitlines()
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    if not indents:
        return text
    cut = min(indents)
    return "\n".join(l[cut:] if len(l) >= cut else l.lstrip() for l in lines)


_parser = XLAParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)
