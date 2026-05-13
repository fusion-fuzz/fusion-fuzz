"""
Unit tests for core/driver.py — BaseDriver and DockerDriver.

subprocess is fully mocked — no Docker, no real processes.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from core.driver import BaseDriver, DockerDriver, ExecutionResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_config(project="testproj", timeout=5, crash_patterns=None):
    return {
        "project_name": project,
        "execution": {
            "timeout": timeout,
            "command": "echo {seed_path}",
        },
        "analysis": {
            "crash_patterns": crash_patterns or ["SUMMARY:", "Segmentation fault"],
        },
    }


def _fake_seed(content="print('hello')", seed_id="abc123"):
    seed = MagicMock()
    seed.id = seed_id
    seed.content = content
    seed.metadata = {}
    return seed


# ---------------------------------------------------------------------------
# BaseDriver — _run_command
# ---------------------------------------------------------------------------

class TestBaseDriverRunCommand(unittest.TestCase):

    def setUp(self):
        self.driver = BaseDriver(_base_config())

    def test_success(self):
        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.return_value = (b"hello\n", b"")
            proc.returncode = 0
            mock_popen.return_value = proc
            rc, stdout, stderr = self.driver._run_command("echo hello")
        self.assertEqual(rc, 0)
        self.assertIn("hello", stdout)

    def test_timeout_returns_124(self):
        import subprocess
        with patch("subprocess.Popen") as mock_popen:
            proc = MagicMock()
            proc.communicate.side_effect = subprocess.TimeoutExpired("cmd", 5)
            proc.pid = 999
            mock_popen.return_value = proc
            with patch("os.killpg"), patch("os.getpgid", return_value=999):
                rc, stdout, stderr = self.driver._run_command("sleep 100")
        self.assertEqual(rc, 124)
        self.assertEqual(stderr, "TIMEOUT")

    def test_exception_returns_1(self):
        with patch("subprocess.Popen", side_effect=OSError("no such file")):
            rc, stdout, stderr = self.driver._run_command("notacommand")
        self.assertEqual(rc, 1)
        self.assertIn("no such file", stderr)


# ---------------------------------------------------------------------------
# BaseDriver — _check_crash
# ---------------------------------------------------------------------------

class TestBaseDriverCheckCrash(unittest.TestCase):

    def setUp(self):
        self.driver = BaseDriver(_base_config(crash_patterns=["SUMMARY:", "Segmentation fault"]))

    def test_pattern_in_stderr(self):
        self.assertTrue(self.driver._check_crash("", "SUMMARY: AddressSanitizer: ...", 1))

    def test_pattern_in_stdout(self):
        self.assertTrue(self.driver._check_crash("Segmentation fault", "", 139))

    def test_no_pattern(self):
        self.assertFalse(self.driver._check_crash("normal output", "warning: unused variable", 0))

    def test_empty_patterns_never_crashes(self):
        driver = BaseDriver(_base_config(crash_patterns=[]))
        self.assertFalse(driver._check_crash("", "anything", 1))


# ---------------------------------------------------------------------------
# BaseDriver — extract_crash_signature
# ---------------------------------------------------------------------------

class TestBaseDriverSignature(unittest.TestCase):

    def setUp(self):
        self.driver = BaseDriver(_base_config())

    def test_asan_in_stderr(self):
        stderr = "==1==ERROR: ...\nSUMMARY: AddressSanitizer: heap-use-after-free"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("heap-use-after-free", sig)

    def test_asan_in_stdout(self):
        stdout = "SUMMARY: AddressSanitizer: stack-buffer-overflow"
        sig = self.driver.extract_crash_signature(stdout, "", 1)
        self.assertIn("stack-buffer-overflow", sig)

    def test_no_asan_returns_none(self):
        sig = self.driver.extract_crash_signature("fine output", "", 0)
        self.assertIsNone(sig)


# ---------------------------------------------------------------------------
# BaseDriver — execute()
# ---------------------------------------------------------------------------

class TestBaseDriverExecute(unittest.TestCase):

    def setUp(self):
        self.driver = BaseDriver(_base_config(crash_patterns=["SUMMARY:"]))

    def _mock_run(self, rc=0, stdout="ok", stderr=""):
        return patch.object(self.driver, "_run_command", return_value=(rc, stdout, stderr))

    def test_execute_returns_execution_result(self):
        with self._mock_run():
            result = self.driver.execute(_fake_seed())
        self.assertIsInstance(result, ExecutionResult)

    def test_no_crash_on_clean_output(self):
        with self._mock_run(rc=0, stdout="ok", stderr=""):
            result = self.driver.execute(_fake_seed())
        self.assertFalse(result.crashed)
        self.assertIsNone(result.signature)

    def test_crash_detected_on_pattern(self):
        with self._mock_run(rc=1, stderr="SUMMARY: AddressSanitizer: heap-use-after-free"):
            result = self.driver.execute(_fake_seed())
        self.assertTrue(result.crashed)
        self.assertIsNotNone(result.signature)
        self.assertIn("heap-use-after-free", result.signature)

    def test_result_stores_command(self):
        with self._mock_run():
            result = self.driver.execute(_fake_seed())
        self.assertIsNotNone(result.command)

    def test_timeout_not_crash(self):
        """A timeout (rc=124) with no crash pattern should not be flagged as a crash."""
        with self._mock_run(rc=124, stderr="TIMEOUT"):
            result = self.driver.execute(_fake_seed())
        self.assertFalse(result.crashed)

    def test_workdir_cleaned_up(self):
        """Temp workdir must be removed even when execution succeeds."""
        created_dirs = []
        original_mkdtemp = tempfile.mkdtemp

        def tracking_mkdtemp(**kwargs):
            d = original_mkdtemp(**kwargs)
            created_dirs.append(d)
            return d

        with patch("tempfile.mkdtemp", side_effect=tracking_mkdtemp):
            with self._mock_run():
                self.driver.execute(_fake_seed())

        for d in created_dirs:
            self.assertFalse(os.path.exists(d), f"Workdir {d} was not cleaned up")


# ---------------------------------------------------------------------------
# DockerDriver — _ensure_container
# ---------------------------------------------------------------------------

class TestDockerDriverEnsureContainer(unittest.TestCase):
    """
    Test the _ensure_container logic without actually calling Docker.
    All subprocess calls are mocked.
    """

    def _make_docker_driver(self, config=None):
        """Build a minimal DockerDriver subclass and instance without real __init__ side-effects."""

        class FakeDockerDriver(DockerDriver):
            container_name = "ffl-fake"
            container_image = "ffl-fake-image"
            file_ext = "txt"

            def _build_exec_cmd(self, container_path, seed):
                return f"docker exec {self.container_name} cat {container_path}"

        cfg = config or _base_config()

        with patch("subprocess.run"), patch("os.makedirs"):
            # bypass real DockerDriver.__init__ side-effects (makedirs, _ensure_container)
            instance = FakeDockerDriver.__new__(FakeDockerDriver)
            instance.config = cfg
            instance.project_name = cfg["project_name"]
            instance.timeout = 5
            instance.container_name = "ffl-fake"
            instance.container_image = "ffl-fake-image"
            instance.file_ext = "txt"
            instance.project_root = "/fake/root"
            instance.host_tmp = tempfile.mkdtemp()
        return instance

    def tearDown(self):
        pass

    def test_does_nothing_when_container_running(self):
        driver = self._make_docker_driver()
        with patch("subprocess.run") as mock_run:
            # First call (docker exec … true) succeeds → container is up
            mock_run.return_value = MagicMock(returncode=0)
            driver._ensure_container()
        # docker rm -f and docker run should NOT have been called
        calls_str = [str(c) for c in mock_run.call_args_list]
        self.assertFalse(any("rm" in s for s in calls_str))
        self.assertFalse(any("run" in s for s in calls_str))

    def test_restarts_when_container_missing(self):
        import subprocess
        driver = self._make_docker_driver()
        with patch("subprocess.run") as mock_run:
            # First call (docker exec … true) fails → container not running
            mock_run.side_effect = [
                subprocess.CalledProcessError(1, "docker exec"),
                MagicMock(returncode=0),   # docker rm -f
                MagicMock(returncode=0),   # docker run
            ]
            driver._ensure_container()
        calls_str = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(any("rm" in s for s in calls_str))
        self.assertTrue(any("run" in s for s in calls_str))

    def test_verify_mounts_false_triggers_restart(self):
        import subprocess
        driver = self._make_docker_driver()
        with patch("subprocess.run") as mock_run:
            # exec … true succeeds but _verify_container_mounts returns False
            mock_run.return_value = MagicMock(returncode=0)
            with patch.object(driver, "_verify_container_mounts", return_value=False):
                driver._ensure_container()
        calls_str = [str(c) for c in mock_run.call_args_list]
        self.assertTrue(any("rm" in s for s in calls_str))


# ---------------------------------------------------------------------------
# DockerDriver — execute()
# ---------------------------------------------------------------------------

class TestDockerDriverExecute(unittest.TestCase):

    def _make_driver(self):
        class FakeDockerDriver(DockerDriver):
            container_name = "ffl-fake"
            container_image = "ffl-fake-image"
            file_ext = "rs"

            def _build_exec_cmd(self, container_path, seed):
                return f"docker exec {self.container_name} rustc {container_path}"

        cfg = _base_config(crash_patterns=["SUMMARY:", "panicked at"])
        with patch("subprocess.run"), patch("os.makedirs"):
            instance = FakeDockerDriver.__new__(FakeDockerDriver)
            instance.config = cfg
            instance.project_name = "fake"
            instance.timeout = 5
            instance.container_name = "ffl-fake"
            instance.container_image = "ffl-fake-image"
            instance.file_ext = "rs"
            instance.project_root = "/fake/root"
            instance.host_tmp = tempfile.mkdtemp()
        return instance

    def test_host_temp_file_written_and_removed(self):
        driver = self._make_driver()
        seed = _fake_seed(content="fn main(){}", seed_id="seed1")

        with patch.object(driver, "_run_command", return_value=(0, "ok", "")):
            driver.execute(seed)

        leftover = os.path.join(driver.host_tmp, "seed1.rs")
        self.assertFalse(os.path.exists(leftover), "Host temp file was not cleaned up")

    def test_returns_execution_result(self):
        driver = self._make_driver()
        with patch.object(driver, "_run_command", return_value=(0, "", "")):
            result = driver.execute(_fake_seed())
        self.assertIsInstance(result, ExecutionResult)

    def test_crash_detected(self):
        driver = self._make_driver()
        with patch.object(driver, "_run_command",
                          return_value=(101, "", "thread 'main' panicked at 'overflow'")):
            result = driver.execute(_fake_seed())
        self.assertTrue(result.crashed)

    def test_no_crash_on_clean_run(self):
        driver = self._make_driver()
        with patch.object(driver, "_run_command", return_value=(0, "compiled ok", "")):
            result = driver.execute(_fake_seed())
        self.assertFalse(result.crashed)

    def test_write_failure_returns_error_result(self):
        driver = self._make_driver()
        # Make host_tmp a path that can't be written to
        driver.host_tmp = "/nonexistent_path_xyz"
        result = driver.execute(_fake_seed())
        self.assertEqual(result.return_code, 1)
        self.assertFalse(result.crashed)

    def test_post_cleanup_called(self):
        driver = self._make_driver()
        cleanup_calls = []
        driver._post_cleanup = lambda ht, s: cleanup_calls.append(ht)
        with patch.object(driver, "_run_command", return_value=(0, "", "")):
            driver.execute(_fake_seed(seed_id="xyz"))
        self.assertEqual(len(cleanup_calls), 1)

    def test_dynamic_file_ext(self):
        """_get_file_ext override is respected in the temp file name."""
        driver = self._make_driver()
        driver._get_file_ext = lambda seed: "go"
        seed = _fake_seed(seed_id="myseed")
        written_paths = []

        original_open = open
        def tracking_open(path, *args, **kwargs):
            if "myseed" in str(path):
                written_paths.append(path)
            return original_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=tracking_open):
            with patch.object(driver, "_run_command", return_value=(0, "", "")):
                driver.execute(seed)

        self.assertTrue(any(str(p).endswith(".go") for p in written_paths),
                        f"Expected .go extension, got paths: {written_paths}")


if __name__ == "__main__":
    unittest.main()
