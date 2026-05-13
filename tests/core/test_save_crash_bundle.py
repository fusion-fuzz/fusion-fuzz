"""
Unit tests for FusionFuzzLoop._save_crash_bundle.

File naming convention tested:
  test.<ext>     — original reproducer
  min.<ext>      — minimized reproducer placeholder (same content initially)
  test.out       — combined stdout + stderr
  test.sh        — reproducing shell command (executable)
  parent_a.<ext> — parent A program
  parent_b.<ext> — parent B program
  README.md      — human-readable bug report

All filesystem I/O goes to a real tempdir cleaned up after each test;
no real driver, no real fuzzing loop.
"""

import os
import re
import sys
import stat
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.fusion import Seed
from core.driver import ExecutionResult
from core.orchestrator import FusionFuzzLoop


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(stderr="", stdout="", return_code=1):
    r = ExecutionResult(
        return_code=return_code,
        stdout=stdout,
        stderr=stderr,
        time=0.1,
        crashed=True,
        signature=None,
    )
    r.command = None
    return r


def _make_orchestrator(project_name, corpus, tmp_cwd):
    ffl = FusionFuzzLoop.__new__(FusionFuzzLoop)
    ffl.config = {
        "project_name": project_name,
        "execution": {"command": f"run {{seed_path}}"},
    }
    ffl.project_name = project_name
    ffl.corpus = corpus
    ffl.original_cwd = tmp_cwd
    ffl.unique_crashes = set()
    return ffl


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _crash_dir(tmp_cwd, project_name, sig, seed_id=None):
    # Folder name is the sanitized signature only — no seed-ID suffix.
    return os.path.join(tmp_cwd, "output", "bugs", project_name, sig)


# ---------------------------------------------------------------------------
# 1. Reproducer files — test.<ext> and min.<ext>
# ---------------------------------------------------------------------------

class TestReproducerFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, project, ext):
        seed = Seed(content="crash_code()", metadata={})
        seed.id = "aabb1122"
        ffl = _make_orchestrator(project, [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "sig")
        d = _crash_dir(self.tmp, project, "sig", seed.id)
        return d

    def _check(self, project, ext):
        d = self._run(project, ext)
        self.assertTrue(os.path.exists(os.path.join(d, f"test{ext}")),
                        f"test{ext} missing for {project}")
        self.assertTrue(os.path.exists(os.path.join(d, f"min{ext}")),
                        f"min{ext} missing for {project}")
        self.assertEqual(_read(os.path.join(d, f"test{ext}")), "crash_code()")
        self.assertEqual(_read(os.path.join(d, f"min{ext}")), "crash_code()")

    def test_python(self):   self._check("cpython", ".py")
    def test_php(self):      self._check("php",     ".phpt")
    def test_rust(self):     self._check("rust",    ".rs")
    def test_mlir(self):     self._check("mlir",    ".mlir")
    def test_swift(self):    self._check("swift",   ".swift")
    def test_wgsl(self):     self._check("naga",    ".wgsl")
    def test_go(self):       self._check("go",      ".go")
    def test_fallback_txt(self): self._check("unknown", ".txt")

    def test_no_reproduce_file_created(self):
        """Legacy 'reproduce.*' name must NOT be created."""
        d = self._run("cpython", ".py")
        legacy = [f for f in os.listdir(d) if f.startswith("reproduce")]
        self.assertEqual(legacy, [], f"Unexpected legacy files: {legacy}")


# ---------------------------------------------------------------------------
# 2. test.out — combined stdout + stderr
# ---------------------------------------------------------------------------

class TestOutputFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def test_combined_output_written(self):
        seed = Seed(content="x", metadata={})
        seed.id = "out00001"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(stderr="err line", stdout="out line"), "sig_out")
        d = _crash_dir(self.tmp, "cpython", "sig_out", seed.id)
        content = _read(os.path.join(d, "test.out"))
        self.assertIn("err line", content)
        self.assertIn("out line", content)

    def test_empty_output_file_still_created(self):
        seed = Seed(content="x", metadata={})
        seed.id = "out00002"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "sig_empty")
        d = _crash_dir(self.tmp, "cpython", "sig_empty", seed.id)
        self.assertTrue(os.path.exists(os.path.join(d, "test.out")))


# ---------------------------------------------------------------------------
# 3. test.sh — reproducing command
# ---------------------------------------------------------------------------

class TestCommandScript(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_bundle(self, sig="sig_sh"):
        seed = Seed(content="x", metadata={})
        seed.id = "sh000001"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), sig)
        return _crash_dir(self.tmp, "cpython", sig, seed.id)

    def test_test_sh_created(self):
        d = self._make_bundle()
        self.assertTrue(os.path.exists(os.path.join(d, "test.sh")))

    def test_no_reproduce_sh_created(self):
        d = self._make_bundle()
        self.assertFalse(os.path.exists(os.path.join(d, "reproduce.sh")),
                         "Legacy reproduce.sh must not be created")

    def test_test_sh_is_executable(self):
        d = self._make_bundle()
        mode = os.stat(os.path.join(d, "test.sh")).st_mode
        self.assertTrue(mode & stat.S_IXUSR)

    def test_test_sh_references_test_filename(self):
        d = self._make_bundle()
        content = _read(os.path.join(d, "test.sh"))
        self.assertIn("test", content)
        self.assertNotIn("reproduce", content)


# ---------------------------------------------------------------------------
# 4. Parent files — parent_a.<ext>, parent_b.<ext>
# ---------------------------------------------------------------------------

class TestParentFiles(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def _parent(self, pid, content, meta=None):
        s = Seed(content=content, metadata=meta or {})
        s.id = pid
        return s

    def test_both_parents_written(self):
        pa = self._parent("pa000001", "# parent A")
        pb = self._parent("pb000002", "# parent B")
        child = Seed(content="# child", metadata={"parents": [pa.id, pb.id]})
        child.id = "child001"
        ffl = _make_orchestrator("cpython", [pa, pb], self.tmp)
        ffl._save_crash_bundle(child, _make_result(), "sig_parents")
        d = _crash_dir(self.tmp, "cpython", "sig_parents", child.id)
        self.assertEqual(_read(os.path.join(d, "parent_a.py")), pa.content)
        self.assertEqual(_read(os.path.join(d, "parent_b.py")), pb.content)

    def test_no_parent_files_when_no_parents(self):
        child = Seed(content="x", metadata={})
        child.id = "orphan01"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(child, _make_result(), "sig_orphan")
        d = _crash_dir(self.tmp, "cpython", "sig_orphan", child.id)
        self.assertEqual([f for f in os.listdir(d) if f.startswith("parent_")], [])

    def test_missing_parent_skipped_gracefully(self):
        child = Seed(content="x", metadata={"parents": ["ghost1", "ghost2"]})
        child.id = "ghost001"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(child, _make_result(), "sig_ghost")
        d = _crash_dir(self.tmp, "cpython", "sig_ghost", child.id)
        self.assertEqual([f for f in os.listdir(d) if f.startswith("parent_")], [])

    def test_partial_parents(self):
        pa = self._parent("real_pa", "# real A")
        child = Seed(content="x", metadata={"parents": ["real_pa", "missing_pb"]})
        child.id = "partial1"
        ffl = _make_orchestrator("cpython", [pa], self.tmp)
        ffl._save_crash_bundle(child, _make_result(), "sig_partial")
        d = _crash_dir(self.tmp, "cpython", "sig_partial", child.id)
        files = os.listdir(d)
        self.assertIn("parent_a.py", files)
        self.assertNotIn("parent_b.py", files)


# ---------------------------------------------------------------------------
# 5. README.md — content and format
# ---------------------------------------------------------------------------

class TestReadme(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def _readme(self, project, child, corpus, stderr="err", stdout="out"):
        ffl = _make_orchestrator(project, corpus, self.tmp)
        result = _make_result(stderr=stderr, stdout=stdout)
        ffl._save_crash_bundle(child, result, "readme_sig")
        d = _crash_dir(self.tmp, project, "readme_sig", child.id)
        return _read(os.path.join(d, "README.md"))

    def test_no_report_md_created(self):
        """Legacy report.md must NOT be created."""
        child = Seed(content="x", metadata={}); child.id = "rpt00001"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(child, _make_result(), "sig_rpt")
        d = _crash_dir(self.tmp, "cpython", "sig_rpt", child.id)
        self.assertFalse(os.path.exists(os.path.join(d, "report.md")))
        self.assertTrue(os.path.exists(os.path.join(d, "README.md")))

    def test_title_present(self):
        child = Seed(content="x", metadata={}); child.id = "rdm00001"
        md = self._readme("cpython", child, [])
        self.assertIn("Fusion-Fuzz Bug Report", md)

    def test_metadata_line(self):
        child = Seed(content="x", metadata={}); child.id = "rdm00002"
        md = self._readme("cpython", child, [])
        self.assertIn(child.id, md)
        self.assertIn("readme_sig", md)

    def test_code_block_present(self):
        child = Seed(content="print('hi')", metadata={}); child.id = "rdm00003"
        md = self._readme("cpython", child, [])
        self.assertIn("The following code:", md)
        self.assertIn("print('hi')", md)

    def test_output_block_present(self):
        child = Seed(content="x", metadata={}); child.id = "rdm00004"
        md = self._readme("cpython", child, [], stderr="boom", stdout="ok")
        self.assertIn("Resulted in this output:", md)
        self.assertIn("boom", md)
        self.assertIn("ok", md)

    def test_to_reproduce_section(self):
        child = Seed(content="x", metadata={}); child.id = "rdm00005"
        md = self._readme("cpython", child, [])
        self.assertIn("To reproduce:", md)

    def test_footer_link(self):
        child = Seed(content="x", metadata={}); child.id = "rdm00006"
        md = self._readme("cpython", child, [])
        self.assertIn("Fusion-Fuzz", md)

    def test_bug_corpus_parent_label(self):
        pa = Seed(content="# A", metadata={
            "type": "bug_corpus", "source_project": "cpython", "source_name": "issue_999"})
        pa.id = "bugcorp1"
        child = Seed(content="x", metadata={"parents": [pa.id]}); child.id = "rdm00007"
        md = self._readme("cpython", child, [pa])
        self.assertIn("Bug corpus", md)
        self.assertIn("cpython", md)
        self.assertIn("issue_999", md)

    def test_project_seed_parent_label(self):
        pa = Seed(content="# A", metadata={"type": "python", "identifier": "corpus/foo.py"})
        pa.id = "projseed1"
        child = Seed(content="x", metadata={"parents": [pa.id]}); child.id = "rdm00008"
        md = self._readme("cpython", child, [pa])
        self.assertIn("Project seed", md)
        self.assertIn("corpus/foo.py", md)

    def test_ghost_parent_ids_in_readme(self):
        child = Seed(content="x", metadata={"parents": ["ghost1", "ghost2"]}); child.id = "rdm00009"
        md = self._readme("cpython", child, [])
        self.assertIn("ghost1", md)
        self.assertIn("ghost2", md)


# ---------------------------------------------------------------------------
# 6. PHP / phpt extraction
# ---------------------------------------------------------------------------

PHPT_CONTENT = """\
--TEST--
Fused seed_a + seed_b
--INI--
precision=12
--FILE--
<?php
echo "hello";
var_dump(1 + 1);
--EXPECT--
hello
int(2)
"""

PHPT_NO_FILE_SECTION = """\
--TEST--
No file section here
--EXPECT--
nothing
"""


class TestFolderNaming(unittest.TestCase):
    """Folder name = sanitized signature only, no seed-ID suffix."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_seed_id_in_folder_name(self):
        seed = Seed(content="x", metadata={})
        seed.id = "deadbeef"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "my_sig")
        bugs_root = os.path.join(self.tmp, "output", "bugs", "cpython")
        folders = os.listdir(bugs_root)
        self.assertEqual(folders, ["my_sig"],
                         f"Expected ['my_sig'], got {folders}")

    def test_same_signature_reuses_folder(self):
        """Two different seeds with identical signatures must write to the same folder."""
        seed1 = Seed(content="crash A", metadata={}); seed1.id = "aaaa0001"
        seed2 = Seed(content="crash B", metadata={}); seed2.id = "bbbb0002"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed1, _make_result(), "dup_sig")
        ffl._save_crash_bundle(seed2, _make_result(), "dup_sig")
        bugs_root = os.path.join(self.tmp, "output", "bugs", "cpython")
        self.assertEqual(os.listdir(bugs_root), ["dup_sig"])

    def test_different_signatures_get_separate_folders(self):
        seed1 = Seed(content="x", metadata={}); seed1.id = "aaaa0001"
        seed2 = Seed(content="y", metadata={}); seed2.id = "bbbb0002"
        ffl = _make_orchestrator("cpython", [], self.tmp)
        ffl._save_crash_bundle(seed1, _make_result(), "sig_one")
        ffl._save_crash_bundle(seed2, _make_result(), "sig_two")
        bugs_root = os.path.join(self.tmp, "output", "bugs", "cpython")
        self.assertCountEqual(os.listdir(bugs_root), ["sig_one", "sig_two"])


class TestPhptExtraction(unittest.TestCase):
    """_extract_display_code must strip phpt boilerplate for PHP projects."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil; shutil.rmtree(self.tmp, ignore_errors=True)

    def test_readme_shows_php_code_not_phpt(self):
        """README.md code block must contain only the --FILE-- section."""
        seed = Seed(content=PHPT_CONTENT, metadata={})
        seed.id = "php00001"
        ffl = _make_orchestrator("php", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(stderr="Assertion failed"), "sig_php")
        d = _crash_dir(self.tmp, "php", "sig_php", seed.id)
        md = _read(os.path.join(d, "README.md"))

        # PHP code should be present
        self.assertIn('echo "hello"', md)
        self.assertIn("var_dump(1 + 1)", md)
        # phpt boilerplate should NOT be in the code block
        self.assertNotIn("--TEST--", md)
        self.assertNotIn("--INI--", md)
        self.assertNotIn("--EXPECT--", md)
        self.assertNotIn("Fused seed_a", md)

    def test_readme_lang_hint_is_php_not_phpt(self):
        """The fenced code block in README.md must open with ```php, not ```phpt."""
        seed = Seed(content=PHPT_CONTENT, metadata={})
        seed.id = "php00002"
        ffl = _make_orchestrator("php", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "sig_php2")
        d = _crash_dir(self.tmp, "php", "sig_php2", seed.id)
        md = _read(os.path.join(d, "README.md"))
        self.assertIn("```php\n", md)
        self.assertNotIn("```phpt", md)

    def test_test_phpt_keeps_full_content(self):
        """test.phpt must still contain the complete phpt including all sections."""
        seed = Seed(content=PHPT_CONTENT, metadata={})
        seed.id = "php00003"
        ffl = _make_orchestrator("php", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "sig_php3")
        d = _crash_dir(self.tmp, "php", "sig_php3", seed.id)
        raw = _read(os.path.join(d, "test.phpt"))
        self.assertIn("--TEST--", raw)
        self.assertIn("--FILE--", raw)
        self.assertIn("--EXPECT--", raw)
        self.assertIn("--INI--", raw)

    def test_fallback_to_full_content_when_no_file_section(self):
        """If --FILE-- is absent, the full phpt content is used as-is."""
        seed = Seed(content=PHPT_NO_FILE_SECTION, metadata={})
        seed.id = "php00004"
        ffl = _make_orchestrator("php", [], self.tmp)
        ffl._save_crash_bundle(seed, _make_result(), "sig_php4")
        d = _crash_dir(self.tmp, "php", "sig_php4", seed.id)
        md = _read(os.path.join(d, "README.md"))
        # Falls back to the whole content
        self.assertIn("No file section here", md)


if __name__ == "__main__":
    unittest.main()
