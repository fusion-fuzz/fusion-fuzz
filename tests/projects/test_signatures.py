"""
Unit tests for extract_crash_signature() in every project driver.

These are pure-function tests — no Docker, no subprocess, no filesystem access.
Each test passes raw stderr/stdout strings and asserts the returned signature label.
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ---------------------------------------------------------------------------
# Helpers — build minimal driver instances without triggering Docker/setup
# ---------------------------------------------------------------------------

def _make_config(project_name):
    return {
        "project_name": project_name,
        "execution": {"timeout": 5},
        "analysis": {"crash_patterns": ["SUMMARY:", "Segmentation fault", "Fatal Python error",
                                         "internal compiler error", "panic:", "INTERNAL PANIC"]},
    }

def _load_driver(project_name, driver_class_name):
    """Import a driver module and return an uninitialised instance (bypass __init__)."""
    import importlib.util
    driver_path = os.path.join("projects", project_name, "driver.py")
    if not os.path.exists(driver_path):
        raise unittest.SkipTest(f"projects/{project_name}/driver.py not found — skipping")
    module_name = f"ffl_{project_name}_driver"
    spec = importlib.util.spec_from_file_location(module_name, driver_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod

    # Patch subprocess and os.makedirs so __init__ doesn't touch Docker or filesystem
    with patch("subprocess.run"), patch("subprocess.Popen"), patch("os.makedirs"):
        spec.loader.exec_module(mod)
        cls = getattr(mod, driver_class_name)
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance.config = _make_config(project_name)
        instance.project_name = project_name
        instance.timeout = 5
        instance.container_name = f"ffl-{project_name}"
        instance.host_tmp = "/tmp/ffl_test"
        instance.project_root = "/fake/project/root"
    return instance


# ---------------------------------------------------------------------------
# CPython
# ---------------------------------------------------------------------------

class TestCPythonSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("cpython", "CPythonDriver")

    def test_asan(self):
        stderr = "==12345==ERROR: AddressSanitizer: heap-use-after-free\nSUMMARY: AddressSanitizer: heap-use-after-free /src/Objects/dictobject.c:1234"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("heap-use-after-free", sig)

    def test_fatal_python_error(self):
        stderr = "Fatal Python error: Segmentation fault\nThread 0x00007f...\n"
        sig = self.driver.extract_crash_signature("", stderr, 139)
        self.assertEqual(sig, "Segmentation fault")

    def test_assertion_failed(self):
        stderr = "python: Objects/dictobject.c:1503: insertdict: Assertion `value != NULL` failed."
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion", sig)
        self.assertIn("value != NULL", sig)

    def test_bus_error(self):
        sig = self.driver.extract_crash_signature("", "Bus error (core dumped)", 135)
        self.assertEqual(sig, "Bus error")

    def test_segfault(self):
        sig = self.driver.extract_crash_signature("", "Segmentation fault (core dumped)", 139)
        self.assertEqual(sig, "Segmentation fault")

    def test_no_crash_returns_none(self):
        sig = self.driver.extract_crash_signature("hello world", "", 0)
        self.assertIsNone(sig)


# ---------------------------------------------------------------------------
# Go
# ---------------------------------------------------------------------------

class TestGoSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("go", "GoDriver")

    def test_ice(self):
        combined = "internal compiler error: out of memory in (*Func).growStack\n"
        sig = self.driver.extract_crash_signature("", combined, 2)
        self.assertIn("internal compiler error", sig)

    def test_panic(self):
        combined = "panic: interface conversion: interface {} is nil, not *types.Basic\n"
        sig = self.driver.extract_crash_signature("", combined, 2)
        self.assertIn("compiler panic", sig)
        self.assertIn("interface conversion", sig)

    def test_fatal_error(self):
        combined = "fatal error: runtime: out of memory\n"
        sig = self.driver.extract_crash_signature("", combined, 2)
        self.assertIn("compiler fatal", sig)

    def test_segfault(self):
        sig = self.driver.extract_crash_signature("", "Segmentation fault", 139)
        self.assertIn("Segmentation fault", sig)

    def test_panic_in_stdout(self):
        # Go compiler writes to stdout sometimes
        sig = self.driver.extract_crash_signature("panic: unexpected nil\n", "", 2)
        self.assertIn("compiler panic", sig)


# ---------------------------------------------------------------------------
# Lean
# ---------------------------------------------------------------------------

class TestLeanSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("lean", "LeanDriver")
        cls.driver._lean_bin = "/fake/lean"

    def test_internal_panic(self):
        stderr = "INTERNAL PANIC: failed to find declaration 'Foo.bar' in environment\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("failed to find declaration", sig)

    def test_rust_thread_panic(self):
        stderr = "thread 'main' panicked at 'index out of bounds: the len is 0 but the index is 0', src/lean.rs:42\n"
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("index out of bounds", sig)

    def test_segfault(self):
        sig = self.driver.extract_crash_signature("", "Segmentation fault", 139)
        self.assertEqual(sig, "Segmentation fault")

    def test_internal_panic_in_stdout(self):
        stdout = "INTERNAL PANIC: type mismatch\n"
        sig = self.driver.extract_crash_signature(stdout, "", 1)
        self.assertIn("type mismatch", sig)


# ---------------------------------------------------------------------------
# PHP
# ---------------------------------------------------------------------------

class TestPHPSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("php", "PHPDriver")

    def test_assertion(self):
        stderr = "Assertion: value != NULL failed at zend_hash.c:800\n"
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion", sig)

    def test_asan_summary(self):
        stderr = "SUMMARY: AddressSanitizer: use-after-poison\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("SUMMARY", sig)

    def test_assertion_in_stdout(self):
        stdout = "Assertion: zval_gc_flags != 0 failed\n"
        sig = self.driver.extract_crash_signature(stdout, "", 134)
        self.assertIn("Assertion", sig)


# ---------------------------------------------------------------------------
# Clang
# ---------------------------------------------------------------------------

class TestClangSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("clang", "ClangDriver")
        cls.driver.container_name = "ffl-clang"

    def test_asan(self):
        stderr = "SUMMARY: AddressSanitizer: stack-buffer-overflow\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("ASAN", sig)
        self.assertIn("stack-buffer-overflow", sig)

    def test_assertion(self):
        stderr = "clang: /llvm/lib/IR/Value.cpp:123: void llvm::Value::replaceAllUsesWith: Assertion `New->getType() == getType()' failed."
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion", sig)

    def test_llvm_error(self):
        stderr = "LLVM ERROR: out of memory\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("LLVM ERROR", sig)
        self.assertIn("out of memory", sig)

    def test_stack_dump(self):
        stderr = "Stack dump:\n0.\tProgram arguments: clang test.c\n1.\t<eof> parser at end of file\n\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("Stack dump", sig)

    def test_segfault(self):
        sig = self.driver.extract_crash_signature("", "Segmentation fault (core dumped)", 139)
        self.assertEqual(sig, "Segmentation fault")

    def test_aborted(self):
        sig = self.driver.extract_crash_signature("", "Aborted (core dumped)", 134)
        self.assertEqual(sig, "Aborted")


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

class TestSwiftSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("swift", "SwiftDriver")
        cls.driver.container_name = "ffl-swift"

    def test_assertion(self):
        stderr = "Assertion failed: (pointer != nullptr), function emitAddressAtScope, file SILGen/SILGenExpr.cpp, line 1234.\n"
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion failed", sig)

    def test_request_evaluation(self):
        stderr = "1. While evaluating request TypeCheckFunctionBodyRequest\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("While evaluating request", sig)

    def test_asan_via_base(self):
        stderr = "SUMMARY: AddressSanitizer: heap-buffer-overflow\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("heap-buffer-overflow", sig)

    def test_no_crash_returns_none(self):
        sig = self.driver.extract_crash_signature("normal output", "warning: unused var", 0)
        self.assertIsNone(sig)


if __name__ == "__main__":
    unittest.main()
