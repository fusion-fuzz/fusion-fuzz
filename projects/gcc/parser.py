"""
projects/gcc/parser.py — turn C/C++ files into seeds for the GCC adapter.

The dataflow analysis is clang's, imported rather than copied: GCC and
clang consume the same two languages, and `CFastDataflow` is a
line-co-occurrence heuristic over C identifiers with nothing
clang-specific in it.

Emitting `variables`/`dataflows` is not optional. ClangFusionStrategy —
which the GCC entry in core/fusion.py's STRATEGY_REGISTRY reuses — begins
its bridge construction with

    if not dataflow1 or not dataflow2: return code1, code2

so a seed without those keys makes dataflow fusion a silent no-op: the run
looks healthy, the children come back unfused. load_corpus backfills them
for any corpus that arrived without them.
"""

import importlib.util
import os

from core.parser import BaseParser

# core/parser.py and main.py both load project parsers by file path, so
# there is no package context for a relative import. The absolute form
# works because FFL runs from the repo root.
try:
    from projects.clang.parser import CFastDataflow
except ImportError:  # pragma: no cover - direct-load fallback
    _clang_parser = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "clang", "parser.py")
    _spec = importlib.util.spec_from_file_location("ffl_clang_parser", _clang_parser)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    CFastDataflow = _mod.CFastDataflow

# NOTE the case handling: GCC's testsuite uses ".C" for C++ (all of
# g++.old-deja does), and ".c" for C. Lowercasing the extension before
# comparing — the obvious thing to do — collapses those two into one and
# hands every ".C" test to `gcc`, which rejects it as C. setup.py already
# normalises collected seeds to ".c"/".cc" by directory, so this is the
# second line of defence, for seeds that arrive by any other route.
_CXX_EXTS = ('.cc', '.cpp', '.cxx', '.hpp', '.ii')


class GCCParser(BaseParser):
    # .i / .ii are preprocessed sources; GCC's testsuite has plenty and they
    # compile standalone with no include path, which makes them unusually
    # good seeds.
    # ".C" is listed explicitly because endswith() is case-sensitive and
    # ".c" therefore does not match it. Headers are deliberately absent:
    # they are not translation units.
    extensions = ['.c', '.C', '.cc', '.cpp', '.cxx', '.i', '.ii']
    seed_type = 'c'  # per-seed override below, by extension

    def parse_content(self, content, filename=""):
        raw_ext = os.path.splitext(filename)[1]
        # ".C" is C++; every other spelling is matched case-insensitively.
        is_cxx = raw_ext == ".C" or raw_ext.lower() in _CXX_EXTS
        ext = ".cc" if raw_ext == ".C" else raw_ext.lower()
        variables, dataflows = CFastDataflow().analyze(content)
        return {
            # "c"/"cpp" are also the keys core/dryrun.py's _COLLECTORS uses
            # to pick a metadata collector, so this string is what routes
            # GCC seeds to ClangMetadataCollector (has_declaration).
            "type":       "cpp" if is_cxx else "c",
            "is_dejagnu": "{ dg-" in content,
            "extension":  ext,
            "variables":  variables,
            "dataflows":  dataflows,
        }

    def load_corpus(self, db_path):
        """Load, backfilling dataflow metadata for any seed missing it.

        setup.py builds corpus.db through this same parser, so on a normal
        install every seed already carries these keys and this loop does
        nothing. It exists for corpora built by some other route, where the
        absence would otherwise turn dataflow fusion into a silent no-op
        rather than an error.
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


_parser = GCCParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing GCC parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
