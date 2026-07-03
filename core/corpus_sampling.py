import json
import random
import sqlite3
import uuid


def save_subset(seeds, db_path):
    """Persist a seed subset to a sqlite DB using the same (identifier, content,
    metadata) schema as project corpus.db files, so it can be reloaded later
    without recomputing selection."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seeds (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            identifier TEXT UNIQUE,
            content    TEXT,
            metadata   TEXT
        )
    """)
    for s in seeds:
        identifier = s.metadata.get("filename") or s.id or str(uuid.uuid4())[:8]
        conn.execute(
            "INSERT OR REPLACE INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)",
            (identifier, s.content, json.dumps(s.metadata)),
        )
    conn.commit()
    conn.close()


def load_subset(db_path):
    """Load a previously saved seed subset (see save_subset) into Seed objects."""
    from core.fusion import Seed

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT identifier, content, metadata FROM seeds").fetchall()
    conn.close()
    return [
        Seed(content=r[1], metadata={**json.loads(r[2]), "filename": r[0]})
        for r in rows
    ]


def _shingles(content, k=3):
    """Word k-gram shingles of seed content, used as a cheap structural fingerprint."""
    toks = content.split()
    if len(toks) < k:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def select_diverse_seeds(seeds, n, k=3, seed_rng=None):
    """Greedy farthest-point sampling: pick n seeds that are pairwise dissimilar,
    best effort, using Jaccard similarity over word k-gram shingles.

    O(len(seeds) * n): each round only compares remaining candidates against the
    most recently picked seed, updating a running best-similarity-so-far per
    candidate rather than rescanning the whole selected set.
    """
    if n >= len(seeds):
        return list(seeds)

    rng = seed_rng or random
    fingerprints = [_shingles(s.content, k) for s in seeds]

    remaining = list(range(len(seeds)))
    start = rng.randrange(len(remaining))
    picked_idx = remaining.pop(start)
    selected = [seeds[picked_idx]]

    # best_sim[i] = highest similarity seen so far between candidate i and any selected seed
    best_sim = {i: _jaccard(fingerprints[i], fingerprints[picked_idx]) for i in remaining}

    while len(selected) < n and remaining:
        # Pick the candidate least similar to anything already selected
        next_i = min(remaining, key=lambda i: best_sim[i])
        remaining.remove(next_i)
        selected.append(seeds[next_i])

        for i in remaining:
            sim = _jaccard(fingerprints[i], fingerprints[next_i])
            if sim > best_sim[i]:
                best_sim[i] = sim

    return selected
