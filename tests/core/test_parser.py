"""
Unit tests for core/parser.py — BaseParser.

Tests DB operations using a real temporary SQLite file (no mocking needed —
SQLite is a stdlib dependency, not an external service).
Tests collect_seeds() using a real temporary directory of fixture files.
"""
import sys
import os
import json
import sqlite3
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.parser import BaseParser


# ---------------------------------------------------------------------------
# Concrete subclass used throughout the tests
# ---------------------------------------------------------------------------

class SimpleParser(BaseParser):
    extensions = [".txt"]
    seed_type = "text"

    def parse_content(self, content, filename=""):
        return {"word_count": len(content.split())}


# ---------------------------------------------------------------------------
# _save_to_db / load_corpus
# ---------------------------------------------------------------------------

class TestBaseParserDB(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        # Point the parser at the temp dir so corpus.db lands there
        self.parser = SimpleParser.__new__(SimpleParser)
        self.parser._project_dir = self.tmp_dir
        self.parser.extensions = [".txt"]
        self.parser.seed_type = "text"
        self.db_path = os.path.join(self.tmp_dir, "corpus.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _seeds(self, n=3):
        return [
            {"identifier": f"seed_{i}.txt",
             "content": f"content {i}",
             "metadata": {"type": "text", "word_count": 2, "filename": f"seed_{i}.txt"}}
            for i in range(n)
        ]

    # --- _save_to_db ---

    def test_save_creates_db_file(self):
        self.parser._save_to_db(self._seeds())
        self.assertTrue(os.path.exists(self.db_path))

    def test_save_returns_db_path(self):
        result = self.parser._save_to_db(self._seeds())
        self.assertEqual(result, self.db_path)

    def test_save_correct_row_count(self):
        self.parser._save_to_db(self._seeds(5))
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        conn.close()
        self.assertEqual(count, 5)

    def test_save_deduplication(self):
        """Inserting the same identifier twice should not raise and should not double-count."""
        seeds = self._seeds(3)
        self.parser._save_to_db(seeds)
        self.parser._save_to_db(seeds)   # second insert: all duplicates
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM seeds").fetchone()[0]
        conn.close()
        self.assertEqual(count, 3)

    def test_save_schema_has_identifier_unique(self):
        self.parser._save_to_db(self._seeds(1))
        conn = sqlite3.connect(self.db_path)
        info = conn.execute("PRAGMA table_info(seeds)").fetchall()
        conn.close()
        col_names = [row[1] for row in info]
        self.assertIn("identifier", col_names)
        self.assertIn("content", col_names)
        self.assertIn("metadata", col_names)

    def test_save_metadata_is_valid_json(self):
        self.parser._save_to_db(self._seeds(2))
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("SELECT metadata FROM seeds").fetchall()
        conn.close()
        for (meta_str,) in rows:
            parsed = json.loads(meta_str)   # must not raise
            self.assertIn("type", parsed)

    def test_save_empty_list_returns_db_path(self):
        result = self.parser._save_to_db([])
        self.assertEqual(result, self.db_path)

    # --- load_corpus ---

    def test_load_returns_list(self):
        self.parser._save_to_db(self._seeds(3))
        loaded = self.parser.load_corpus(self.db_path)
        self.assertIsInstance(loaded, list)
        self.assertEqual(len(loaded), 3)

    def test_load_keys(self):
        self.parser._save_to_db(self._seeds(1))
        loaded = self.parser.load_corpus(self.db_path)
        self.assertIn("filename", loaded[0])
        self.assertIn("content", loaded[0])
        self.assertIn("metadata", loaded[0])

    def test_load_metadata_is_dict(self):
        self.parser._save_to_db(self._seeds(2))
        loaded = self.parser.load_corpus(self.db_path)
        for item in loaded:
            self.assertIsInstance(item["metadata"], dict)

    def test_load_missing_db_returns_empty(self):
        result = self.parser.load_corpus("/nonexistent/path/corpus.db")
        self.assertEqual(result, [])

    def test_roundtrip_content_preserved(self):
        seeds = [{"identifier": "hello.txt", "content": "Hello, world!",
                  "metadata": {"type": "text", "word_count": 2, "filename": "hello.txt"}}]
        self.parser._save_to_db(seeds)
        loaded = self.parser.load_corpus(self.db_path)
        self.assertEqual(loaded[0]["content"], "Hello, world!")


# ---------------------------------------------------------------------------
# collect_seeds — uses a real temporary directory
# ---------------------------------------------------------------------------

class TestBaseParserCollectSeeds(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.parser = SimpleParser(os.path.join(self.tmp_dir, "fake_parser.py"))

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write(self, name, content="hello world"):
        path = os.path.join(self.tmp_dir, name)
        with open(path, "w") as f:
            f.write(content)
        return path

    def test_collects_matching_extension(self):
        self._write("a.txt", "one two three")
        self._write("b.txt", "four five")
        db_path = self.parser.collect_seeds(self.tmp_dir)
        loaded = self.parser.load_corpus(db_path)
        names = {item["filename"] for item in loaded}
        self.assertIn("a.txt", names)
        self.assertIn("b.txt", names)

    def test_ignores_non_matching_extension(self):
        self._write("a.txt", "valid")
        self._write("b.rs", "should be ignored")
        db_path = self.parser.collect_seeds(self.tmp_dir)
        loaded = self.parser.load_corpus(db_path)
        self.assertEqual(len(loaded), 1)

    def test_returns_none_for_missing_path(self):
        result = self.parser.collect_seeds("/does/not/exist")
        self.assertIsNone(result)

    def test_metadata_type_set(self):
        self._write("seed.txt", "hello world")
        db_path = self.parser.collect_seeds(self.tmp_dir)
        loaded = self.parser.load_corpus(db_path)
        self.assertEqual(loaded[0]["metadata"]["type"], "text")

    def test_metadata_filename_set(self):
        self._write("seed.txt", "hello world")
        db_path = self.parser.collect_seeds(self.tmp_dir)
        loaded = self.parser.load_corpus(db_path)
        self.assertEqual(loaded[0]["metadata"]["filename"], "seed.txt")

    def test_custom_parse_content_called(self):
        self._write("seed.txt", "one two three four")
        db_path = self.parser.collect_seeds(self.tmp_dir)
        loaded = self.parser.load_corpus(db_path)
        self.assertEqual(loaded[0]["metadata"]["word_count"], 4)

    def test_empty_directory_returns_db_path(self):
        db_path = self.parser.collect_seeds(self.tmp_dir)
        self.assertIsNotNone(db_path)
        loaded = self.parser.load_corpus(db_path)
        self.assertEqual(loaded, [])

    def test_type_override_from_parse_content(self):
        """If parse_content sets 'type', it should override the class seed_type."""
        class DynamicParser(BaseParser):
            extensions = [".txt"]
            seed_type = "default"
            def parse_content(self, content, filename=""):
                return {"type": "overridden"}

        parser = DynamicParser(os.path.join(self.tmp_dir, "fake.py"))
        self._write("x.txt", "data")
        db = parser.collect_seeds(self.tmp_dir)
        loaded = parser.load_corpus(db)
        self.assertEqual(loaded[0]["metadata"]["type"], "overridden")


if __name__ == "__main__":
    unittest.main()
