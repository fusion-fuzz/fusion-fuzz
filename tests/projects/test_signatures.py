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
        """The signature keeps the "Fatal Python error" prefix rather than
        reducing to the bare signal name: the interpreter's own fatal
        handler firing and a raw SIGSEGV are different failures and should
        not share a bucket."""
        stderr = "Fatal Python error: Segmentation fault\nThread 0x00007f...\n"
        sig = self.driver.extract_crash_signature("", stderr, 139)
        self.assertEqual(sig, "Fatal Python error: Segmentation fault")

    def test_assertion_failed_glibc_format(self):
        """glibc's assert() closes the expression with a single quote.

        This test previously fed ``Assertion `value != NULL` failed`` — a
        backtick on *both* sides — which is not a format any assert()
        produces. It matched the driver's regex because that regex had the
        same mistake, so the test passed while the adapter could not
        recognise a single real assertion failure.
        """
        stderr = ("python: Objects/dictobject.c:1503: insertdict: "
                  "Assertion `value != NULL' failed.")
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion", sig)
        self.assertIn("value != NULL", sig)
        self.assertIn("Objects/dictobject.c:1503", sig)

    def test_assertion_failed_cpython_format(self):
        """CPython's own _PyObject_AssertFailed (Objects/object.c) uses
        double quotes and appends a message — a second spelling the old
        regex also missed."""
        stderr = ('Objects/object.c:275: _Py_NegativeRefcount: '
                  'Assertion "op->ob_refcnt > 0" failed: object has negative ref count')
        sig = self.driver.extract_crash_signature("", stderr, 134)
        self.assertIn("Assertion", sig)
        self.assertIn("op->ob_refcnt > 0", sig)

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


# ---------------------------------------------------------------------------
# Naga
# ---------------------------------------------------------------------------

class TestNagaSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("naga", "NagaDriver")

    def test_rust_panic(self):
        stderr = "thread 'main' panicked at naga/src/valid/mod.rs:123: impossible\n"
        sig = self.driver.extract_crash_signature("", stderr, 101)
        self.assertIn("Rust panic", sig)

    def test_parse_error_is_not_signature(self):
        sig = self.driver.extract_crash_signature("", "Could not parse WGSL:\nerror: expected `;`\n", 1)
        self.assertIsNone(sig)


# ---------------------------------------------------------------------------
# Haskell
# ---------------------------------------------------------------------------

class TestHaskellSignatures(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.driver = _load_driver("haskell", "HaskellDriver")

    def test_ghc_panic(self):
        stderr = "ghc: panic! (the 'impossible' happened)\n  (GHC version 9.14.1)\n  Simplifier ticks exhausted\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("ghc panic", sig)

    def test_ghc_internal_error(self):
        stderr = "GHC internal error: unexpected coercion\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("GHC internal error", sig)

    def test_ghc_internal_error_lowercase(self):
        stderr = "internal error: PAP object entered!\n"
        sig = self.driver.extract_crash_signature("", stderr, 1)
        self.assertIn("GHC internal error", sig)

    def test_segfault(self):
        sig = self.driver.extract_crash_signature("", "Segmentation fault", 139)
        self.assertEqual(sig, "ghc: Segmentation fault")

    def test_no_crash_returns_none(self):
        sig = self.driver.extract_crash_signature("42\n", "", 0)
        self.assertIsNone(sig)


class TestGoSignatureGrouping(unittest.TestCase):
    """The same Go compiler bug must group under one signature no matter
    which function the fused test happened to trigger it in."""

    def setUp(self):
        from projects.go.analyzer import crash_signature
        self.sig = crash_signature

    def _ice(self, pos, symbol, message):
        return (
            f"{pos}: internal compiler error: '{symbol}': {message}\n"
            "\n"
            "goroutine 19 [running]:\n"
            "runtime/debug.Stack()\n"
            "cmd/compile/internal/base.FatalfAt({0x2acbf250?, 0x1093?}, {0x109, 0x1d})\n"
            "cmd/compile/internal/base.Fatalf(...)\n"
            "cmd/compile/internal/ssagen.(*ssafn).Fatalf(0x2f1?, {0x2?, 0x0?})\n"
            "cmd/compile/internal/ssa.(*Func).FatalfWithPos(0x10932ad961c0, {0x1982408?})\n"
            "cmd/compile/internal/ssa.(*Func).Fatalf(...)\n"
            "cmd/compile/internal/ssacompile.checkFunc(0x10932ad961c0)\n"
            "cmd/compile/internal/ssagen.buildssa({0x1a40040, 0x1ceab80}, 0x10932ac3e3c0)\n"
        )

    def test_symbol_prefix_does_not_split_one_bug(self):
        msg = "unknown aux type for LoweredZero"
        sigs = {
            self.sig(self._ice("./main.go:21:2", "main", msg)),
            self.sig(self._ice("./main.go:222:15", "main_dbb0557", msg)),
            self.sig(self._ice("<autogenerated>:1", "(*Bar_b584f86).Get3Vals", msg)),
            self.sig(self._ice(
                "<autogenerated>:1",
                "5614d73739c4cb953979c5b476501cda147b86386834ff88f636587787e0f1e9.FieldByName",
                msg)),
        }
        self.assertEqual(len(sigs), 1, sigs)

    def test_long_generic_instantiation_symbol_is_stripped(self):
        """A generic instantiation symbol runs to hundreds of characters --
        it must not survive into the signature just because it is long."""
        sym = ("EqualMap[go.shape.map[go.shape.struct { main.hi uint64; "
               "main.lo uint64; main.z *uint8 }]struct {},go.shape.map["
               "go.shape.struct { main.hi uint64; main.lo uint64; "
               "main.z *uint8 }]struct {},go.shape.struct { main.hi uint64; "
               "main.lo uint64; main.z *uint8 },go.shape.struct {}]")
        self.assertGreater(len(sym), 250)
        msg = "unknown aux type for LoweredZero"
        self.assertEqual(
            self.sig(self._ice("./main.go:18:15", sym, msg)),
            self.sig(self._ice("./main.go:21:2", "main", msg)))

    def test_reporting_frames_are_skipped(self):
        sig = self.sig(self._ice("./main.go:21:2", "main", "unknown aux type for LoweredZero"))
        self.assertEqual(
            sig, "ICE: unknown aux type for LoweredZero "
                 "[cmd/compile/internal/ssacompile.checkFunc]")

    def test_distinct_messages_still_split(self):
        a = self.sig(self._ice("./main.go:1:1", "f", "unknown aux type for LoweredZero"))
        b = self.sig(self._ice("./main.go:1:1", "f", "unknown aux type for LoweredMove"))
        self.assertNotEqual(a, b)

    def test_decimal_array_length_is_not_a_hash(self):
        """A long decimal number is not a fusion hash.

        Decimal digits are a subset of the hex alphabet, so a hash pattern
        that does not demand an a-f also eats array lengths -- and Go prints
        those in the very messages we group on."""
        a = self._ice("./x.go:4:6", "main",
                      "bad type: struct { b [1214748364700000000]byte }")
        b = self._ice("./x.go:4:6", "main",
                      "bad type: struct { b [1500000000]byte }")
        self.assertIn("1214748364700000000", self.sig(a))
        self.assertNotEqual(self.sig(a), self.sig(b))

    def test_fusion_hash_suffix_is_still_stripped(self):
        """The restriction above must not cost us the real hashes."""
        a = self._ice("./x.go:9:6", "main", "bad type: struct { large_d430158 }")
        b = self._ice("./x.go:9:6", "main", "bad type: struct { large_bd84f3a }")
        self.assertEqual(self.sig(a), self.sig(b))
        self.assertIn("large_HASH", self.sig(a))


if __name__ == "__main__":
    unittest.main()
