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
# Lean
# ---------------------------------------------------------------------------

class TestLeanParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.lean.parser import _parser
        cls.parser = _parser

    def test_imports(self):
        code = "import Mathlib.Algebra.Group.Basic\nimport Lean.Elab.Tactic"
        meta = self.parser.parse_content(code)
        self.assertIn("Mathlib.Algebra.Group.Basic", meta["imports"])
        self.assertIn("Lean.Elab.Tactic", meta["imports"])

    def test_theorems_and_lemmas(self):
        code = "theorem myThm : 1 + 1 = 2 := rfl\nlemma helper : True := trivial"
        meta = self.parser.parse_content(code)
        self.assertIn("myThm", meta["theorems"])
        self.assertIn("helper", meta["theorems"])

    def test_defs(self):
        code = "def foo : Nat := 42\nnoncomputable def bar := 0\nprivate def baz := ()"
        meta = self.parser.parse_content(code)
        self.assertIn("foo", meta["defs"])
        self.assertIn("bar", meta["defs"])
        self.assertIn("baz", meta["defs"])

    def test_structures(self):
        code = "structure Point where\n  x : Float\nclass Functor (f : Type) where"
        meta = self.parser.parse_content(code)
        self.assertIn("Point", meta["structures"])
        self.assertIn("Functor", meta["structures"])

    def test_namespaces(self):
        code = "namespace MyLib\ndef helper := 1\nend MyLib"
        meta = self.parser.parse_content(code)
        self.assertIn("MyLib", meta["namespaces"])

    def test_empty_file(self):
        meta = self.parser.parse_content("")
        for key in ("imports", "defs", "theorems", "structures", "namespaces"):
            self.assertEqual(meta[key], [], f"Expected empty list for {key}")


# ---------------------------------------------------------------------------
# WGSL (Naga and WGSLC share the same parse_content logic)
# ---------------------------------------------------------------------------

class TestWGSLParser(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from projects.naga.parser import _parser
        cls.parser = _parser

    def test_functions(self):
        code = "@vertex\nfn vs_main(in: VertexInput) -> VertexOutput {}\n@fragment\nfn fs_main() -> @location(0) vec4<f32> {}"
        meta = self.parser.parse_content(code)
        self.assertIn("vs_main", meta["functions"])
        self.assertIn("fs_main", meta["functions"])

    def test_structs(self):
        code = "struct VertexInput { @location(0) position: vec3<f32> }\nstruct Uniforms { mvp: mat4x4<f32> }"
        meta = self.parser.parse_content(code)
        self.assertIn("VertexInput", meta["structs"])
        self.assertIn("Uniforms", meta["structs"])

    def test_empty_file(self):
        meta = self.parser.parse_content("")
        self.assertEqual(meta["functions"], [])
        self.assertEqual(meta["structs"], [])


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


if __name__ == "__main__":
    unittest.main()
