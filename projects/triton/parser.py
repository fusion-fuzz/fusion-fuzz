"""
projects/triton/parser.py — turn Triton .mlir tests into seeds.

Beyond the variables/dataflows every fusion strategy consumes, this records
what a Triton test's `// RUN:` line declares, because that is what decides
how the seed can be run at all:

  passes        the pass pipeline. `-tritongpu-pipeline=num-stages=3` *is*
                what the test exercises; the same IR under some other
                pipeline exercises nothing in particular, so the pipeline
                travels with the seed into projects/triton/driver.py.
  aliases       layout alias names (`#blocked = #ttg.blocked<...>`). 209 of
                Triton's tests define at least one and the IR body
                references them, so fusing two modules collides their
                aliases unless the strategy renames them.
  module_attrs  the `module attributes {...}` payload, which carries the
                target (`ttg.target = "cuda:80"`) the passes read.

The dataflow analysis is MLIR's own, imported rather than copied:
triton-opt consumes standard MLIR text, and `%0`-style SSA values are
lexically what the MLIR adapter already handles. Emitting
`variables`/`dataflows` is not optional — a strategy that finds them
missing degrades to a silent no-op, which is what projects/mlir/parser.py
does today.
"""

import importlib.util
import os
import re

from core.parser import BaseParser

# main.py and core/parser.py load project parsers by file path, so there is
# no package context for a relative import. The absolute form works because
# FFL runs from the repo root.
try:
    from projects.triton.analyzer import analyze_seed
except ImportError:  # pragma: no cover - direct-load fallback
    _spec = importlib.util.spec_from_file_location(
        "ffl_triton_analyzer_parser",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    analyze_seed = _mod.analyze_seed


# MLIR SSA values and block arguments: %0, %arg1, %cst_2. These are the
# "variables" the dataflow strategy renames across two modules.
_SSA_DEF_RE = re.compile(r'^\s*(%[\w$.]+)(?:\s*,\s*(%[\w$.]+))*\s*=', re.M)
_SSA_USE_RE = re.compile(r'(%[\w$.]+)')


class MLIRDataflow:
    """Line-co-occurrence dataflow over MLIR SSA values.

    The same heuristic the clang adapter uses for C identifiers, applied to
    `%value` names: two values are related when they appear on the same
    line, which in SSA form means one is an operand of the op defining the
    other. That is exactly the def-use edge the dataflow strategy wants,
    and in MLIR it is available without building a real graph — SSA makes
    every definition textual and unique.
    """

    def analyze(self, content):
        variables, dataflows = [], []
        seen = set()
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            names = _SSA_USE_RE.findall(line)
            if not names:
                continue
            for n in names:
                if n not in seen:
                    seen.add(n)
                    variables.append(n)
            # A definition and its operands on one line.
            if "=" in line and len(names) > 1:
                lhs = names[0]
                for rhs in names[1:]:
                    if rhs != lhs:
                        dataflows.append((lhs, rhs))
        return variables, dataflows


class TritonParser(BaseParser):
    extensions = ['.mlir']
    seed_type = 'mlir'

    def parse_content(self, content, filename=""):
        facts = analyze_seed(content, filename)
        variables, dataflows = MLIRDataflow().analyze(content)
        return {
            # core/dryrun.py's _COLLECTORS keys on this. Triton consumes
            # MLIR, so it shares the collector, the lexicon and the fusion
            # strategies with the MLIR adapter; what differs is the driver
            # and the crash oracle.
            "type":          "mlir",
            "extension":     ".mlir",
            "passes":        facts["passes"],
            "needs_split":   facts["needs_split"],
            "aliases":       facts["aliases"],
            "module_attrs":  facts["module_attrs"],
            "func_names":    facts["func_names"],
            "has_run_line":  facts["has_run_line"],
            "variables":     variables,
            "dataflows":     dataflows,
        }

    def load_corpus(self, db_path):
        """Load, backfilling dataflow metadata for any seed missing it.

        setup.py builds corpus.db through this same parser, so on a normal
        install this loop does nothing. It exists for corpora built by some
        other route, where the absence would turn dataflow fusion into a
        silent no-op rather than an error.
        """
        seeds = super().load_corpus(db_path)
        analyzer = MLIRDataflow()
        for seed in seeds:
            meta = seed.get("metadata") or {}
            if meta.get("dataflows"):
                continue
            variables, dataflows = analyzer.analyze(seed.get("content") or "")
            meta["variables"] = variables
            meta["dataflows"] = dataflows
            seed["metadata"] = meta
        return seeds


_parser = TritonParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing Triton parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
