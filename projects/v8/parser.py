"""
projects/v8/parser.py — turn JavaScript files into seeds.

Beyond the variables/dataflows every fusion strategy consumes, this records
the facts that decide how a V8 seed must be *run*:

  flags          mjsunit tests declare what they need in a leading
                 `// Flags:` comment. They are part of the test, not
                 decoration: a test written around
                 %OptimizeFunctionOnNextCall does nothing without
                 --allow-natives-syntax.
  uses_natives   %-prefixed runtime functions are a *syntax* error without
                 --allow-natives-syntax, so this flag is not optional
                 tuning — the driver has to add it or the seed cannot even
                 parse.
  needs_load     the file `load()`s a sibling script that a fused seed no
                 longer sits next to.

The dataflow analysis is clang's CFastDataflow, imported rather than
copied: it is a line-co-occurrence heuristic over identifiers with nothing
C-specific in it, and JavaScript identifiers are lexically the same shape.
Emitting `variables`/`dataflows` is not optional — a strategy that finds
them missing degrades to a silent no-op.
"""

import importlib.util
import os

from core.parser import BaseParser

# main.py and core/parser.py load project parsers by file path, so there is
# no package context for a relative import. The absolute form works because
# FFL runs from the repo root.
try:
    from projects.clang.parser import CFastDataflow
except ImportError:  # pragma: no cover - direct-load fallback
    _clang_parser = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "clang", "parser.py")
    _spec = importlib.util.spec_from_file_location("ffl_clang_parser", _clang_parser)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CFastDataflow = _mod.CFastDataflow

try:
    from projects.v8.analyzer import analyze_seed
except ImportError:  # pragma: no cover - direct-load fallback
    _spec = importlib.util.spec_from_file_location(
        "ffl_v8_analyzer_parser",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    analyze_seed = _mod.analyze_seed


class V8Parser(BaseParser):
    extensions = ['.js', '.mjs']
    seed_type = 'javascript'

    def parse_content(self, content, filename=""):
        facts = analyze_seed(content, filename)
        variables, dataflows = CFastDataflow().analyze(content)
        return {
            # core/dryrun.py's _COLLECTORS keys on this.
            "type":           "javascript",
            "extension":      ".js",
            "flags":          facts["flags"],
            "uses_natives":   facts["uses_natives"],
            "optimize_score": facts["optimize_score"],
            "memory_score":   facts["memory_score"],
            "uses_worker":    facts["uses_worker"],
            "uses_wasm":      facts["uses_wasm"],
            "needs_load":     facts["needs_load"],
            "variables":      variables,
            "dataflows":      dataflows,
        }

    def load_corpus(self, db_path):
        """Load, backfilling dataflow metadata for any seed missing it.

        setup.py builds corpus.db through this same parser, so on a normal
        install this loop does nothing. It exists for corpora built by some
        other route, where the absence would turn dataflow fusion into a
        silent no-op rather than an error.
        """
        seeds = super().load_corpus(db_path)
        analyzer = CFastDataflow()
        for seed in seeds:
            meta = seed.get("metadata") or {}
            if meta.get("dataflows"):
                continue
            variables, dataflows = analyzer.analyze(seed.get("content") or "")
            meta["variables"] = variables
            meta["dataflows"] = dataflows
            seed["metadata"] = meta
        return seeds


_parser = V8Parser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing V8 parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
