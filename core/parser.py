import os
import sqlite3
import json


class BaseParser:
    """
    Base class for project seed parsers. Handles directory scanning and
    SQLite corpus management. Subclasses need only set extensions/seed_type
    and implement parse_content().

    Usage in projects/<name>/parser.py:

        from core.parser import BaseParser

        class MyParser(BaseParser):
            extensions = ['.rs']
            seed_type = 'rust'

            def parse_content(self, content, filename=""):
                return {"functions": re.findall(r'fn (\\w+)', content)}

        _parser = MyParser(__file__)
        def collect_seeds(source_path): return _parser.collect_seeds(source_path)
        def load_corpus(db_path): return _parser.load_corpus(db_path)
    """

    extensions: list = []
    seed_type: str = ""

    def __init__(self, parser_file: str):
        """parser_file must be __file__ of the subclass module."""
        self._project_dir = os.path.dirname(os.path.abspath(parser_file))

    def parse_content(self, content: str, filename: str = "") -> dict:
        """
        Override to return language-specific metadata.
        May set 'type' to override the class-level seed_type per seed.
        Must not set 'filename' (handled by the base class).
        """
        return {}

    def collect_seeds(self, source_path: str, blacklist: list = None):
        if not os.path.exists(source_path):
            print(f"Error: Seed source path not found: {source_path}")
            return None

        print(f"Scanning for {self.extensions} files in {source_path}...")
        seed_paths = []
        try:
            for root, _, files in os.walk(source_path):
                for fname in files:
                    if any(fname.endswith(ext) for ext in self.extensions):
                        seed_paths.append(os.path.join(root, fname))
        except OSError as e:
            print(f"Error listing directory: {e}")
            return None

        print(f"Found {len(seed_paths)} seeds. Processing...")
        seeds_output = []
        skipped = 0
        for file_path in seed_paths:
            seed_filename = os.path.relpath(file_path, source_path)
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if blacklist and any(term in content for term in blacklist):
                    skipped += 1
                    continue
                metadata = self.parse_content(content, seed_filename)
                metadata.setdefault("type", self.seed_type)
                metadata["filename"] = seed_filename
                seeds_output.append({
                    "identifier": seed_filename,
                    "content": content,
                    "metadata": metadata,
                })
            except Exception:
                pass

        if skipped:
            print(f"Skipped {skipped} seeds matching blacklist.")
        return self._save_to_db(seeds_output)

    def _save_to_db(self, seeds: list):
        try:
            db_path = os.path.join(self._project_dir, "corpus.db")
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS seeds (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identifier TEXT UNIQUE,
                    content TEXT,
                    metadata TEXT
                )
            """)
            pruned = pruned_identifiers(db_path)
            count = 0
            for seed in seeds:
                if seed["identifier"] in pruned:
                    continue
                try:
                    cursor.execute(
                        "INSERT INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)",
                        (seed["identifier"], seed["content"], json.dumps(seed["metadata"])),
                    )
                    count += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            conn.close()
            print(f"Saved {count} seeds to {db_path}")
            return db_path
        except Exception as e:
            print(f"Error saving to corpus: {e}")
            return None

    def load_corpus(self, db_path: str) -> list:
        if not os.path.exists(db_path):
            return []
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT identifier, content, metadata FROM seeds")
        rows = cursor.fetchall()
        conn.close()
        pruned = pruned_identifiers(db_path)
        return [
            {"filename": r[0], "content": r[1], "metadata": json.loads(r[2])}
            for r in rows if r[0] not in pruned
        ]


PRUNED_TABLE = "pruned_seeds"


def pruned_identifiers(db_path: str) -> set:
    """Identifiers a project's prune pass has rejected (e.g. seeds the
    target itself won't accept — see projects/flang/prune_corpus.py).

    Kept as a table of its own rather than by deleting rows, because the
    seed sources are re-scanned on every `--setup` and re-injected by
    `--bug-corpus`: deletion alone is undone on the next start, which is
    how a pruned corpus quietly grew back to its unfiltered size.
    """
    if not os.path.exists(db_path):
        return set()
    try:
        conn = sqlite3.connect(db_path)
        try:
            rows = conn.execute(f"SELECT identifier FROM {PRUNED_TABLE}").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return set()
    return {r[0] for r in rows}


def record_pruned(db_path: str, identifiers, reason: str = "") -> int:
    """Mark `identifiers` as pruned and drop their rows. Returns the number
    newly recorded."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS {PRUNED_TABLE} ("
            "identifier TEXT PRIMARY KEY, reason TEXT)"
        )
        before = conn.execute(f"SELECT COUNT(*) FROM {PRUNED_TABLE}").fetchone()[0]
        conn.executemany(
            f"INSERT OR IGNORE INTO {PRUNED_TABLE} (identifier, reason) VALUES (?, ?)",
            ((i, reason) for i in identifiers),
        )
        conn.executemany("DELETE FROM seeds WHERE identifier = ?",
                         ((i,) for i in identifiers))
        after = conn.execute(f"SELECT COUNT(*) FROM {PRUNED_TABLE}").fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return after - before
