"""
projects/rust/parser.py — turn .rs files into seeds.

The previous version recorded only `imports`/`functions`/`structs`, none
of which any strategy reads. What fusion actually consumes is recorded
here instead:

  variables/dataflows  RustFusionStrategy builds its cross-seed connection
                       from statements, but core/dryrun.py's
                       RustMetadataCollector and the shared rename path
                       both want these; a seed without them degrades the
                       lightweight path to a no-op.
  features/crate_attrs `#![feature(...)]` and other inner attributes are
                       crate-level and must be hoisted to the top of a
                       fused file — an inner attribute anywhere else is a
                       hard error, so a fusion that just concatenates two
                       feature-gated tests cannot compile.
  edition              tests declare the edition they need; 2015 and 2024
                       differ enough (`dyn`, `async`, raw identifiers) that
                       compiling one under the other is a parse error
                       rather than a test of anything.
  unsafe_score         drives the driver's sanitizer decisions.
  is_known_bug         `//@ known-bug` marks a test that is *supposed* to
                       ICE, so a hit on one is a filed bug, not a finding.
"""

import importlib.util
import os
import re

from core.parser import BaseParser

# main.py and core/parser.py load project parsers by file path, so there is
# no package context for a relative import.
try:
    from projects.clang.parser import CFastDataflow
except ImportError:  # pragma: no cover - direct-load fallback
    _clang = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "clang", "parser.py")
    _spec = importlib.util.spec_from_file_location("ffl_clang_parser", _clang)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CFastDataflow = _mod.CFastDataflow

try:
    from projects.rust.analyzer import analyze_seed
except ImportError:  # pragma: no cover - direct-load fallback
    _spec = importlib.util.spec_from_file_location(
        "ffl_rust_analyzer_parser",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    analyze_seed = _mod.analyze_seed


_USE_RE = re.compile(r'^\s*use\s+([^;]+);', re.M)
_FN_RE = re.compile(r'fn\s+([a-zA-Z0-9_]+)\s*\(')
_STRUCT_RE = re.compile(r'struct\s+([a-zA-Z0-9_]+)')


class RustParser(BaseParser):
    extensions = ['.rs']
    seed_type = 'rust'

    def parse_content(self, content, filename=""):
        facts = analyze_seed(content, filename)
        variables, dataflows = CFastDataflow().analyze(content)
        return {
            # core/dryrun.py's _COLLECTORS keys on this to pick
            # RustMetadataCollector.
            "type":          "rust",
            "extension":     ".rs",
            "edition":       facts["edition"],
            "expectation":   facts["expectation"],
            "compile_flags": facts["compile_flags"],
            "features":      facts["features"],
            "crate_attrs":   facts["crate_attrs"],
            "no_std":        facts["no_std"],
            "no_main":       facts["no_main"],
            "has_main":      facts["has_main"],
            "unsafe_score":  facts["unsafe_score"],
            "is_known_bug":  facts["is_known_bug"],
            "variables":     variables,
            "dataflows":     dataflows,
            # Not read by any strategy today, but cheap and pinned by
            # tests/projects/test_parsers.py — the same argument
            # core/dryrun.py makes for its unread per-language fields.
            "imports":       _USE_RE.findall(content),
            "functions":     _FN_RE.findall(content),
            "structs":       _STRUCT_RE.findall(content),
        }

    def load_corpus(self, db_path):
        """Load, backfilling dataflow metadata for any seed missing it.

        setup.py builds corpus.db through this same parser, so on a normal
        install this does nothing. It covers corpora built by some other
        route, where the absence would silently degrade fusion rather than
        raise.
        """
        seeds = super().load_corpus(db_path)
        analyzer = CFastDataflow()
        for seed in seeds:
            meta = seed.get("metadata") or {}
            if meta.get("dataflows"):
                continue
            meta["variables"], meta["dataflows"] = analyzer.analyze(
                seed.get("content") or "")
            seed["metadata"] = meta
        return seeds


_parser = RustParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing Rust parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
