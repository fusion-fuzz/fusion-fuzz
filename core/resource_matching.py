"""
core/resource_matching.py — Producer-Consumer Guided Fusion

Not every pair of seeds is worth fusing. Meaningful fusion needs an
asymmetric relationship between the two seeds: one must contribute a
semantic resource (a type it instantiates, a symbol it defines) and the
other must expose a context capable of consuming it (a matching type
reference, an unresolved symbol, an import). Two seeds that share nothing
producible/consumable mostly yield a fusion that's syntactically glued
together but semantically inert.

This module extracts a lightweight produce/consume "resource profile" per
seed from metadata already collected during dry-run/preprocessing
(core/dryrun.py — var_types, struct_names, functions, imports, ...), and
scores how compatible two profiles are. It intentionally reuses whatever
metadata a project's collector already populates rather than requiring a
new per-language collector, so it degrades gracefully (falling back to
raw-token overlap) for projects that haven't been dry-run yet.

core/coverage.py's PairwiseCoverageMatrix uses this to bias parent
selection toward producer/consumer-compatible pairs instead of pure
random sampling, while keeping random selection as a fallback so fuzzing
never stalls on sparse metadata.
"""

import re
from dataclasses import dataclass, field
from typing import Set

# Fields written by BaseMetadataCollector subclasses (core/dryrun.py) that
# represent "produced" resources — things this seed defines/instantiates
# that another seed's context could plausibly refer to.
_PRODUCE_KEYS = ("struct_names", "structs", "classes", "functions", "fn_signatures")

# Fields representing "consumption sites" — things this seed references
# without necessarily defining, i.e. slots that a bridged-in resource
# from another seed could plausibly fill.
_CONSUME_KEYS = ("types_used", "imports", "undefined_refs")

# Fields representing typed variable bindings. A variable of type T is
# both a produce signal (this seed can hand out a T) and a consume signal
# (this seed has a slot that already expects a T, so another T-shaped
# value could be substituted in).
_VAR_TYPE_KEYS = ("var_types", "dynamic_types")

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


@dataclass
class ResourceProfile:
    produces: Set[str] = field(default_factory=set)
    consumes: Set[str] = field(default_factory=set)


def _base_symbol(raw: str) -> str:
    """Normalize a raw type/import/signature string to a leaf identifier
    so profiles line up across languages with different qualifier syntax
    (Rust `&mut Foo`, Go `*pkg.Foo`, Python `pkg.mod.Foo`, C++ `const Foo&`)."""
    if not raw:
        return ""
    s = str(raw).strip()
    s = s.rsplit(".", 1)[-1]          # dotted/module-qualified names
    s = s.rsplit("::", 1)[-1]         # Rust/C++ path-qualified names
    m = re.search(r"[A-Za-z_][A-Za-z0-9_]*", s)
    return m.group(0) if m else ""


def extract_resource_profile(seed) -> ResourceProfile:
    """Derive a produce/consume resource profile for one seed.

    Uses precomputed dry-run metadata when available (cheap: just set
    lookups). Falls back to a raw identifier-token overlap when a seed has
    no such metadata yet (e.g. fuzzing without --dry-run), so the matching
    signal is never simply absent — just weaker.
    """
    meta = seed.metadata or {}
    produces: Set[str] = set()
    consumes: Set[str] = set()

    for key in _VAR_TYPE_KEYS:
        for t in (meta.get(key) or {}).values():
            sym = _base_symbol(t)
            if sym:
                produces.add(sym)
                consumes.add(sym)

    for key in _PRODUCE_KEYS:
        for name in (meta.get(key) or []):
            sym = _base_symbol(name)
            if sym:
                produces.add(sym)

    for key in _CONSUME_KEYS:
        for name in (meta.get(key) or []):
            sym = _base_symbol(name)
            if sym:
                consumes.add(sym)

    if not produces and not consumes:
        tokens = set(_IDENT_RE.findall(seed.content or ""))
        produces |= tokens
        consumes |= tokens

    return ResourceProfile(produces=produces, consumes=consumes)


def compatibility_score(profile_a: ResourceProfile, profile_b: ResourceProfile) -> int:
    """How many resources A produces that B can consume, plus vice versa.
    0 means neither seed has anything the other could plausibly use."""
    return (
        len(profile_a.produces & profile_b.consumes)
        + len(profile_b.produces & profile_a.consumes)
    )


def is_compatible(profile_a: ResourceProfile, profile_b: ResourceProfile) -> bool:
    return compatibility_score(profile_a, profile_b) > 0
