"""
projects/cuda/parser.py — CUDA (.cu/.cuh) seed collection for FusionFuzz.

CUDA is C++ with attributes (`__global__`, `__device__`, `__shared__`),
the `<<<grid, block>>>` launch syntax and a few builtins, so the seed
model is the clang adapter's: identifiers grouped by line co-occurrence
as the coarse dataflow signal the fusion strategies use to pick bridge
variables. The C/C++ dataflow class is imported from projects/clang so
the two adapters cannot drift apart.

Seeds are collected by projects/cuda/setup.py into projects/cuda/seeds/
(clang's CodeGenCUDA/SemaCUDA/Driver lit tests with the lit-only
`#include "Inputs/cuda.h"` line removed, and NVIDIA's cuda-samples), so
that a corpus rebuild never touches the source checkouts.
"""

import importlib.util
import os

from core.parser import BaseParser


def _cfast_dataflow():
    """projects/clang/parser.py's CFastDataflow, loaded by path: `projects`
    is not a package, and the class is the same for CUDA."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(os.path.dirname(here), "clang", "parser.py")
    spec = importlib.util.spec_from_file_location("ffl_clang_parser_for_cuda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CFastDataflow


class CUDAParser(BaseParser):
    extensions = ['.cu', '.cuh']
    seed_type = 'cuda'

    def parse_content(self, content, filename=""):
        variables, dataflows = _cfast_dataflow()().analyze(content)
        return {
            "type": "cuda",
            "extension": ".cu",
            "variables": variables,
            "dataflows": dataflows,
            "has_kernel": "__global__" in content,
            "has_launch": "<<<" in content,
            "line_count": len(content.splitlines()),
        }


_parser = CUDAParser(__file__)


def collect_seeds(source_path, blacklist=None):
    return _parser.collect_seeds(source_path, blacklist=blacklist)


def load_corpus(db_path):
    return _parser.load_corpus(db_path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    collect_seeds(os.path.join(here, "seeds"))
