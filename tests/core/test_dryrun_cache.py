"""
core/dryrun.py — the --pre-analysis cache contract.

Everything here exists because the cache failed silently three separate
times in one working session, each time costing throughput rather than
correctness, and each time invisible in the logs:

  * `has_declaration` was added to the Clang collector, but every seed
    already carried `pre_analysis_done: True`, so the field was never
    written and every candidate pair re-scanned both seeds forever.
  * `states_of_interest` — an older key holding maximum-only state points —
    stayed in the corpus while the code read `most_complex_states`, so half
    the analysis on disk was for a field nobody looked at.
  * seeds injected by --bug-corpus carry `type: bug_corpus`, which names no
    language, so the state-point analysis skipped them entirely. They were
    half the clang corpus and re-analysed on every single fusion.

The invariant these tests defend: *if the current --pre-analysis would
write something a seed does not already have, that seed gets re-analysed.*
Both the key-level check and the version stamp serve it — keys catch new
fields, the version catches a field whose meaning changed under a name
that did not.
"""

import json
import os
import sqlite3
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.dryrun import (  # noqa: E402
    OBSOLETE_METADATA_KEYS,
    PRE_ANALYSIS_KEYS,
    PRE_ANALYSIS_VERSION,
    BaseMetadataCollector,
    ClangMetadataCollector,
    GenericMetadataCollector,
    _metadata_is_fresh,
    get_collector,
    run_dryrun_with_metadata,
    update_seed_metadata_in_db,
)
from core.fusion import Seed  # noqa: E402


def _fresh_meta(**overrides):
    """Metadata a seed would carry straight out of a current pre-analysis."""
    meta = {
        "type": "c",
        "filename": "s.c",
        "pre_analysis_done": True,
        "pre_analysis_version": PRE_ANALYSIS_VERSION,
        "has_declaration": False,
    }
    meta.update({k: [] for k in PRE_ANALYSIS_KEYS})
    meta.update(overrides)
    return meta


# ---------------------------------------------------------------------------
# Freshness: which seeds get re-analysed
# ---------------------------------------------------------------------------

def test_complete_metadata_is_fresh():
    assert _metadata_is_fresh(Seed(content="int x;", metadata=_fresh_meta()))


def test_never_analysed_is_not_fresh():
    assert not _metadata_is_fresh(Seed(content="int x;", metadata={"type": "c"}))


@pytest.mark.parametrize("missing", list(PRE_ANALYSIS_KEYS) + ["has_declaration"])
def test_any_missing_key_forces_reanalysis(missing):
    """The `has_declaration` incident: a field added after the corpus was
    built must not be masked by the pre_analysis_done flag."""
    meta = _fresh_meta()
    meta.pop(missing)
    assert not _metadata_is_fresh(Seed(content="int x;", metadata=meta))


def test_stale_version_forces_reanalysis():
    """The version catches a field whose *meaning* changed while its name
    stayed the same — e.g. state points going from 'maximum-count only,
    capped at 25' to 'every line, capped at 100'."""
    assert not _metadata_is_fresh(
        Seed(content="int x;", metadata=_fresh_meta(pre_analysis_version=PRE_ANALYSIS_VERSION - 1)))
    assert not _metadata_is_fresh(
        Seed(content="int x;", metadata=_fresh_meta(pre_analysis_version=None)))


def test_bug_corpus_seed_uses_the_project_language():
    """`bug_corpus` is a provenance marker, not a language. Without the
    project fallback these seeds are considered fresh while carrying no
    analysis at all."""
    bare = Seed(content="int x;", metadata={"type": "bug_corpus", "pre_analysis_done": True,
                                            "pre_analysis_version": PRE_ANALYSIS_VERSION})
    assert not _metadata_is_fresh(bare, fallback_language="clang"), \
        "a bug_corpus seed with no state points must be re-analysed"
    # With no project to fall back on there is no language to analyse for,
    # so it is legitimately considered done.
    assert _metadata_is_fresh(bare, fallback_language=None)


def test_language_with_no_state_analysis_needs_only_collector_keys():
    """A language state_analysis knows nothing about still gets its
    collector's own fields checked, but not the state-point keys."""
    meta = {"type": "unknown_lang", "pre_analysis_done": True,
            "pre_analysis_version": PRE_ANALYSIS_VERSION}
    assert _metadata_is_fresh(Seed(content="x", metadata=meta))


# ---------------------------------------------------------------------------
# What collect() writes
# ---------------------------------------------------------------------------

def test_collect_writes_every_advertised_key():
    seed = Seed(content="struct S { int x; };\nint f(void){ int a = 1; return a; }\n",
                metadata={"type": "c", "filename": "s.c"})
    meta = ClangMetadataCollector().collect(seed, result=None)

    for key in PRE_ANALYSIS_KEYS:
        assert key in meta, f"{key} advertised in PRE_ANALYSIS_KEYS but not written"
    for key in ClangMetadataCollector.provides:
        assert key in meta, f"{key} advertised in provides but not written"
    assert meta["pre_analysis_version"] == PRE_ANALYSIS_VERSION
    assert meta["dryrun_done"] is True


def test_collect_output_is_accepted_as_fresh():
    """The round trip that matters: what collect() writes must satisfy
    _metadata_is_fresh, or every run re-analyses the whole corpus."""
    seed = Seed(content="int f(void){ return 1; }\n", metadata={"type": "c", "filename": "s.c"})
    seed.metadata.update(ClangMetadataCollector().collect(seed, result=None))
    assert _metadata_is_fresh(seed, fallback_language="clang")


def test_collect_uses_fallback_language_for_typeless_seeds():
    seed = Seed(content="int a = 1;\nint b = 2;\n",
                metadata={"type": "bug_corpus", "filename": "b.c"})
    without = GenericMetadataCollector().collect(seed, result=None)
    withfb = GenericMetadataCollector().collect(seed, result=None, fallback_language="clang")
    assert "most_complex_states" not in without
    assert withfb["most_complex_states"], "fallback language should yield state points"
    assert withfb["segment_boundaries"]


def test_state_points_are_recorded_even_when_empty():
    """An explicit [] means "analysed, found none" — dropping the key
    instead would make every fusion re-scan exactly the seeds that always
    come up empty."""
    meta = BaseMetadataCollector().collect(Seed(content="", metadata={"type": "c"}), result=None)
    assert meta.get("most_complex_states") == []


def test_get_collector_falls_back_to_generic():
    assert isinstance(get_collector("c"), ClangMetadataCollector)
    assert isinstance(get_collector("no_such_language"), GenericMetadataCollector)
    assert isinstance(get_collector(""), GenericMetadataCollector)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _db_with_seed(meta):
    path = os.path.join(tempfile.mkdtemp(), "corpus.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE seeds (id INTEGER PRIMARY KEY, identifier TEXT, "
                 "content TEXT, metadata TEXT)")
    conn.execute("INSERT INTO seeds (identifier, content, metadata) VALUES (?,?,?)",
                 ("s.c", "int x;", json.dumps(meta)))
    conn.commit(); conn.close()
    return path


def _read_meta(path):
    conn = sqlite3.connect(path)
    row = conn.execute("SELECT metadata FROM seeds WHERE identifier = 's.c'").fetchone()
    conn.close()
    return json.loads(row[0])


def test_update_merges_rather_than_replaces():
    path = _db_with_seed({"type": "c", "keep_me": 1})
    update_seed_metadata_in_db(path, "s.c", {"rc": 0})
    meta = _read_meta(path)
    assert meta["keep_me"] == 1 and meta["rc"] == 0 and meta["type"] == "c"


def test_obsolete_keys_are_dropped_on_write():
    """`states_of_interest` held maximum-only points; reading it back would
    silently restore the selection scheme it was built with."""
    assert "states_of_interest" in OBSOLETE_METADATA_KEYS
    path = _db_with_seed({"type": "c", "states_of_interest": [{"line_idx": 1}], "keep": 2})
    update_seed_metadata_in_db(path, "s.c", {"most_complex_states": []})
    meta = _read_meta(path)
    assert "states_of_interest" not in meta
    assert meta["keep"] == 2


def test_update_on_missing_identifier_is_a_noop():
    path = _db_with_seed({"type": "c"})
    update_seed_metadata_in_db(path, "does_not_exist.c", {"rc": 0})
    assert "rc" not in _read_meta(path)


# ---------------------------------------------------------------------------
# The pass itself
# ---------------------------------------------------------------------------

class _FakeResult:
    def __init__(self, rc):
        self.return_code = rc
        self.stdout = self.stderr = ""
        self.execution_time = 0.01
        self.crashed = False


class _CountingDriver:
    """Records which seeds actually got executed."""
    executed: list = []

    def __init__(self, rc_for=None):
        self.rc_for = rc_for or {}
        self.timeout = 5

    def execute(self, seed):
        _CountingDriver.executed.append(seed.content)
        return _FakeResult(self.rc_for.get(seed.content, 0))


def _run(seeds, **kw):
    rc_for = kw.pop("rc_for", None) or {}
    _CountingDriver.executed = []
    kept = run_dryrun_with_metadata(
        seeds=seeds, driver_factory=lambda: _CountingDriver(rc_for),
        db_path=None, concurrency=1, timeout=1, project_name="clang", **kw)
    return kept, list(_CountingDriver.executed)


def test_fresh_seeds_are_not_re_executed():
    fresh = Seed(content="int a;", metadata=_fresh_meta())
    stale = Seed(content="int b;", metadata={"type": "c", "filename": "b.c"})
    _, executed = _run([fresh, stale], collect_metadata=True, filter_valid=False)
    assert executed == ["int b;"], f"re-executed a fresh seed: {executed}"


def test_force_re_executes_everything():
    fresh = Seed(content="int a;", metadata=_fresh_meta())
    _, executed = _run([fresh], collect_metadata=True, filter_valid=False, force=True)
    assert executed == ["int a;"]


def test_filter_valid_keeps_only_rc_zero():
    good = Seed(content="good", metadata={"type": "c", "filename": "g.c"})
    bad = Seed(content="bad", metadata={"type": "c", "filename": "b.c"})
    kept, _ = _run([good, bad], collect_metadata=False, filter_valid=True,
                   rc_for={"bad": 1})
    assert [s.content for s in kept] == ["good"]


def test_rc_is_recorded_even_when_not_filtering():
    """rc/dryrun_done are free once the seed has run, so they are recorded
    whether or not the caller asked for validity filtering."""
    seed = Seed(content="x", metadata={"type": "c", "filename": "x.c"})
    kept, executed = _run([seed], collect_metadata=True, filter_valid=False, rc_for={"x": 3})
    assert executed == ["x"]
    assert kept[0].metadata["rc"] == 3
    assert kept[0].metadata["dryrun_done"] is True


def test_neither_flag_executes_nothing():
    """With no filtering and no collection there is nothing to learn from
    running a seed, so the pass is a no-op rather than a wasted execution."""
    seed = Seed(content="x", metadata={"type": "c", "filename": "x.c"})
    kept, executed = _run([seed], collect_metadata=False, filter_valid=False)
    assert executed == []
    assert kept == [seed]
