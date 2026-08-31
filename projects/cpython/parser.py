"""
projects/cpython/parser.py — turn .py files into seeds.

The dataflow analysis below is kept as it was: it walks a real AST rather
than guessing from line co-occurrence the way the C-family parser does,
which is strictly better information and worth keeping.

What changed is everything around it. The previous version reimplemented
seed collection, the sqlite schema and corpus loading from scratch — about
130 lines duplicating core/parser.py's BaseParser, which every other
adapter in this repo subclasses. Two consequences: the metadata keys drift
from what the framework reads, and improvements to BaseParser (the
`identifier` uniqueness handling, the metadata merge) never reach this
project.
"""

import importlib.util
import ast
import os

from core.parser import BaseParser

# main.py loads project parsers by file path, so there is no package
# context for a relative import.
try:
    from projects.cpython.analyzer import analyze_seed
except ImportError:  # pragma: no cover - direct-load fallback
    _spec = importlib.util.spec_from_file_location(
        "ffl_cpython_analyzer_parser",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyzer.py"))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    analyze_seed = _mod.analyze_seed


class PythonFastDataflow:
    """
    Analyzes Python code to extract variables and dataflow groups (interactions) using AST.
    """
    def __init__(self):
        self.variables = set()
        self.interactions = []

    def _get_names(self, node):
        """Recursively collect variable names (ids) from a node."""
        names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                names.add(child.id)
        return names

    def _merge_dataflows(self, groups):
        """
        Merge groups of variables that interact transitively.
        E.g., [a, b] and [b, c] becomes [a, b, c].
        """
        merged = []
        for group in groups:
            group_set = set(group)
            merged_indices = []
            
            # Find all existing groups that overlap with the new group
            for i, m_group in enumerate(merged):
                if not group_set.isdisjoint(m_group):
                    merged_indices.append(i)
            
            if not merged_indices:
                # No overlap, add as new group
                merged.append(group_set)
            else:
                # Overlap found, merge all involved groups
                new_merged_group = group_set
                # Iterate backwards to pop safely
                for i in sorted(merged_indices, reverse=True):
                    new_merged_group.update(merged.pop(i))
                merged.append(new_merged_group)
                
        # Convert sets back to lists for JSON serialization
        return [list(g) for g in merged]

    def analyze(self, code):
        self.variables = set()
        self.interactions = []
        
        try:
            tree = ast.parse(code)
        except Exception:
            # Return empty if syntax error or parse failure
            return [], []

        # Walk the tree to find interactions
        for node in ast.walk(tree):
            # Check statement types where dataflow happens
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign, 
                                 ast.Call, ast.Compare, ast.BinOp, ast.BoolOp, 
                                 ast.Return, ast.Yield)):
                names = self._get_names(node)
                if names:
                    self.variables.update(names)
                    if len(names) > 1:
                        self.interactions.append(list(names))
        
        return list(self.variables), self._merge_dataflows(self.interactions)


class CPythonParser(BaseParser):
    extensions = ['.py']
    seed_type = 'python'

    def parse_content(self, content, filename=""):
        variables, dataflows = PythonFastDataflow().analyze(content)
        facts = analyze_seed(content, filename)
        return {
            # core/dryrun.py's _COLLECTORS and core/state_analysis.py's
            # alias table both accept "python" and map it to the CPython
            # collector / live-variable config.
            "type":       "python",
            "extension":  ".py",
            "is_test":    "test_" in (filename or ""),
            "variables":  variables,
            "dataflows":  dataflows,
            # Containment facts. CPython is the only target here whose
            # seeds are executed, so these are what keep a concurrent loop
            # from binding ports or forking — see projects/cpython/config.yaml.
            "uses_ctypes":          facts["uses_ctypes"],
            "touches_code_objects": facts["touches_code_objects"],
            "uses_network":         facts["uses_network"],
            "uses_subprocess":      facts["uses_subprocess"],
            "writes_fs":            facts["writes_fs"],
            "test_only":            facts["test_only"],
        }

    def load_corpus(self, db_path):
        """Load, backfilling dataflow metadata for any seed missing it.

        setup.py builds corpus.db through this same parser, so on a normal
        install this does nothing. It covers a corpus.db written by the
        previous hand-rolled collector, whose rows predate the containment
        keys.
        """
        seeds = super().load_corpus(db_path)
        analyzer = PythonFastDataflow()
        for seed in seeds:
            meta = seed.get("metadata") or {}
            if meta.get("dataflows") is not None:
                continue
            meta["variables"], meta["dataflows"] = analyzer.analyze(
                seed.get("content") or "")
            seed["metadata"] = meta
        return seeds


_parser = CPythonParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    seeds_dir = os.path.join(script_dir, "seeds")
    print("Executing CPython parser standalone.")
    if os.path.exists(seeds_dir):
        collect_seeds(seeds_dir)
    else:
        print(f"Error: 'seeds' directory not found at {seeds_dir}. Run setup.py first.")
