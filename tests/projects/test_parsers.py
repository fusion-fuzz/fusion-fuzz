"""
Unit tests for parse_content() in every project parser.

Pure-function tests — no filesystem, no SQLite, no subprocess.
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Rust
# ---------------------------------------------------------------------------

class TestRustParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.rust.parser import _parser
        cls.parser = _parser

    def test_functions_extracted(self):
        code = "fn main() {}\nfn helper(x: i32) -> i32 { x }"
        meta = self.parser.parse_content(code)
        self.assertIn("main", meta["functions"])
        self.assertIn("helper", meta["functions"])

    def test_structs_extracted(self):
        code = "struct Point { x: f64, y: f64 }\nstruct Color;"
        meta = self.parser.parse_content(code)
        self.assertIn("Point", meta["structs"])
        self.assertIn("Color", meta["structs"])

    def test_imports_extracted(self):
        code = "use std::io;\nuse std::collections::HashMap;\nfn main() {}"
        meta = self.parser.parse_content(code)
        self.assertIn("std::io", meta["imports"])
        self.assertIn("std::collections::HashMap", meta["imports"])

    def test_empty_file(self):
        meta = self.parser.parse_content("")
        self.assertEqual(meta["functions"], [])
        self.assertEqual(meta["structs"], [])
        self.assertEqual(meta["imports"], [])


# ---------------------------------------------------------------------------
# Swift
# ---------------------------------------------------------------------------

class TestSwiftParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.swift.parser import _parser
        cls.parser = _parser

    def test_has_run_line(self):
        code = "// RUN: %swift-frontend -typecheck %s\nfunc foo() {}"
        meta = self.parser.parse_content(code)
        self.assertTrue(meta["has_frontend_flags"])

    def test_no_run_line(self):
        code = "func bar() -> Int { return 42 }"
        meta = self.parser.parse_content(code)
        self.assertFalse(meta["has_frontend_flags"])


# ---------------------------------------------------------------------------
# GCC — dynamic type based on file extension
# ---------------------------------------------------------------------------

class TestGCCParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.gcc.parser import _parser
        cls.parser = _parser

    def test_c_file_type(self):
        meta = self.parser.parse_content("int main() { return 0; }", filename="test.c")
        self.assertEqual(meta["type"], "c")

    def test_cpp_file_type(self):
        meta = self.parser.parse_content("int main() {}", filename="test.cpp")
        self.assertEqual(meta["type"], "cpp")

    def test_cc_file_type(self):
        meta = self.parser.parse_content("", filename="test.cc")
        self.assertEqual(meta["type"], "cpp")

    def test_hpp_file_type(self):
        meta = self.parser.parse_content("", filename="header.hpp")
        self.assertEqual(meta["type"], "cpp")

    def test_dejagnu_detected(self):
        code = "/* { dg-do compile } */\nint x = 0;"
        meta = self.parser.parse_content(code, filename="test.c")
        self.assertTrue(meta["is_dejagnu"])

    def test_no_dejagnu(self):
        meta = self.parser.parse_content("int main() {}", filename="test.c")
        self.assertFalse(meta["is_dejagnu"])

    def test_extension_stored(self):
        meta = self.parser.parse_content("", filename="foo.cpp")
        self.assertEqual(meta["extension"], ".cpp")


# ---------------------------------------------------------------------------
# MLIR
# ---------------------------------------------------------------------------

class TestMLIRParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.mlir.parser import _parser
        cls.parser = _parser

    def test_dialect_comments_extracted(self):
        code = "// CHECK: arith.addi\n// RUN: mlir-opt\nfunc.func @test() {}"
        meta = self.parser.parse_content(code)
        self.assertTrue(any("arith.addi" in d for d in meta["dialects"]))
        self.assertTrue(any("mlir-opt" in d for d in meta["dialects"]))

    def test_no_comments(self):
        code = "func.func @test() { return }"
        meta = self.parser.parse_content(code)
        self.assertEqual(meta["dialects"], [])


# ---------------------------------------------------------------------------
# Naga / WGSL
# ---------------------------------------------------------------------------

class TestNagaParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.naga.parser import _parser
        cls.parser = _parser

    def test_wgsl_symbols_and_dataflow(self):
        code = (
            "struct Payload { value : i32 }\n"
            "alias MyInt = i32;\n"
            "var<private> counter : i32 = 0;\n"
            "fn main() { let next : i32 = counter + 1; }\n"
        )
        meta = self.parser.parse_content(code, filename="shader.wgsl")
        self.assertEqual(meta["type"], "wgsl")
        self.assertIn("Payload", meta["structs"])
        self.assertIn("MyInt", meta["aliases"])
        self.assertIn("main", meta["functions"])
        self.assertIn("counter", meta["variables"])
        self.assertEqual(meta["var_types"]["counter"], "i32")
        self.assertTrue(meta["has_declaration"])


# ---------------------------------------------------------------------------
# Haskell
# ---------------------------------------------------------------------------

class TestHaskellParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.haskell.parser import _parser
        cls.parser = _parser

    def test_imports(self):
        code = "import Data.IORef\nimport qualified Data.Map as Map\n"
        meta = self.parser.parse_content(code)
        self.assertIn("import Data.IORef", meta["imports"])
        self.assertIn("import qualified Data.Map as Map", meta["imports"])

    def test_toplevel_names(self):
        code = (
            "double :: Int -> Int\n"
            "double x = x * 2\n\n"
            "data Tree = Leaf | Node Tree Int Tree\n\n"
            "class Shape a where\n"
            "  area :: a -> Double\n"
        )
        meta = self.parser.parse_content(code)
        self.assertIn("double", meta["toplevel_names"])
        self.assertIn("Tree", meta["toplevel_names"])
        self.assertIn("Shape", meta["toplevel_names"])
        self.assertNotIn("data", meta["toplevel_names"])

    def test_nullary_bindings(self):
        code = "greeting :: String\ngreeting = \"hello\"\n\ndouble x = x * 2\n"
        meta = self.parser.parse_content(code)
        self.assertIn("greeting", meta["nullary_bindings"])
        self.assertNotIn("double", meta["nullary_bindings"])

    def test_has_main(self):
        code = "main :: IO ()\nmain = print 1\n"
        meta = self.parser.parse_content(code)
        self.assertTrue(meta["has_main"])

    def test_no_main(self):
        meta = self.parser.parse_content("x :: Int\nx = 1\n")
        self.assertFalse(meta["has_main"])

    def test_state_handles(self):
        code = (
            "main :: IO ()\n"
            "main = do\n"
            "  ref <- newIORef (0 :: Int)\n"
            "  writeIORef ref 5\n"
        )
        meta = self.parser.parse_content(code)
        names = [h["name"] for h in meta["state_handles"]]
        self.assertIn("ref", names)
        kinds = {h["name"]: h["kind"] for h in meta["state_handles"]}
        self.assertEqual(kinds["ref"], "ioref")
        self.assertIn("ref", meta["state_used"])

    def test_string_and_char_literals_do_not_confuse_extraction(self):
        code = (
            "greeting :: String\n"
            "greeting = \"data class where\"\n\n"
            "sep :: Char\n"
            "sep = '\\''\n\n"
            "double x = x * 2\n"
        )
        meta = self.parser.parse_content(code)
        self.assertIn("double", meta["toplevel_names"])
        self.assertNotIn("data", meta["toplevel_names"])
        self.assertNotIn("class", meta["toplevel_names"])
        self.assertNotIn("where", meta["toplevel_names"])

    def test_empty_file(self):
        meta = self.parser.parse_content("")
        self.assertEqual(meta["imports"], [])
        self.assertEqual(meta["toplevel_names"], [])
        self.assertEqual(meta["nullary_bindings"], [])
        self.assertFalse(meta["has_main"])
        self.assertEqual(meta["state_handles"], [])


if __name__ == "__main__":
    unittest.main()
