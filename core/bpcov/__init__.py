"""Breakpoint ("first-hit") block coverage for arbitrary target binaries.

No recompilation and no knowledge of the target's language: block leaders
are found by disassembling the executable (extract_blocks.py), a probe
table plus a pre-armed image (int3 at every not-yet-covered leader) is built
(make_table.py), and each execution maps the pre-armed image over its own
text (bpcov.so via LD_PRELOAD) or is traced by bptrace (ptrace; for static
binaries such as Go's toolchain). A probe hit records the block, restores
the original byte and resumes, so the program's behaviour is unchanged and
already-covered code runs at native speed.

CoverageManager is the fuzzer-side half: it wraps commands, collects the
blocks each execution hit, keeps the global covered set, and periodically
rebuilds the tables so covered blocks stop costing anything.
"""
from .manager import CoverageManager  # noqa: F401
