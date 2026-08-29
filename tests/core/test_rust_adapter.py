"""
Tests for the Rust adapter: the crash oracle, the unsafe-Rust signals the
driver keys on, and the crate-structure rules fusion has to respect.

Getting the structure wrong does not lower the valid rate, it zeroes it:
an inner attribute (`#![feature(...)]`) anywhere but the crate root is a
hard error, so a fusion that concatenates two feature-gated tests is
rejected before reaching any compiler path worth testing.
"""

import importlib.util
import os
import random
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from core.fusion import Seed, get_strategies  # noqa: E402


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


analyzer = _load("ffl_rust_analyzer_test", "projects/rust/analyzer.py")


# ---------------------------------------------------------------------------
# The oracle
# ---------------------------------------------------------------------------

ICE = ("error: internal compiler error: unexpected type\n\n"
       "query stack during panic:\n#0 [layout_of] computing layout of `Foo`\n")

DEBUG_ASSERT = (
    "thread 'rustc' panicked at compiler/rustc_middle/src/ty/layout.rs:812:9:\n"
    "assertion failed: !ty.has_infer()\n"
    "stack backtrace:\n   0: rust_begin_unwind\n"
    "   5: rustc_middle::ty::layout::layout_of\n\n"
    "query stack during panic:\n#0 [layout_of] computing layout of `Bar`\n")

LLVM_ASSERT = ("rustc: llvm/lib/IR/Instructions.cpp:1234: void AssertOK(): "
               "Assertion `getOperand(0)->getType() == X' failed.\n")
LLVM_ERROR = "LLVM ERROR: Cannot select: 0x55a1 t23: i64 = X86ISD::CMP\n"
OOM = "memory allocation of 34359738368 bytes failed\n"
STACK = "\nthread 'rustc' has overflowed its stack\nfatal runtime error: stack overflow\n"
RECURSION = "error: reached the recursion limit while instantiating `foo::<...>`\n"
DIAGNOSTIC = "error[E0308]: mismatched types\nerror: aborting due to 1 previous error\n"
ASAN = "SUMMARY: AddressSanitizer: heap-use-after-free /x.rs:5 in main\n"
MIRI = "error: Undefined Behavior: dereferencing a dangling pointer\n"


def test_debug_assertion_is_a_bug_and_groups_on_its_location():
    """A `debug_assert!` firing is only visible because setup.py builds with
    rust.debug-assertions. The file:line groups better than the message,
    which usually embeds the offending type."""
    v = analyzer.classify(DEBUG_ASSERT)
    assert v["is_bug"] and v["kind"] == "panic"
    assert "compiler/rustc_middle/src/ty/layout.rs:812" in v["signature"]


def test_query_name_is_carried_into_the_signature():
    """rustc's query graph names what it was doing when it died. No other
    compiler in this repo gives an equivalent, and it separates two bugs
    that share a panic site."""
    assert analyzer.classify(ICE)["query"] == "layout_of"
    assert "[layout_of]" in analyzer.crash_signature(ICE)


@pytest.mark.parametrize("output,kind", [
    (LLVM_ASSERT, "llvm"), (LLVM_ERROR, "llvm"),
    (ASAN, "sanitizer"), (MIRI, "ub"), (ICE, "ice"),
])
def test_findings_are_reported(output, kind):
    v = analyzer.classify(output)
    assert v["is_bug"] is True and v["kind"] == kind
    assert v["signature"]


@pytest.mark.parametrize("output", [OOM, STACK, RECURSION])
def test_resource_exhaustion_is_not_a_bug(output):
    """Fusion produces deeply nested types as a matter of course, so rustc
    running out of stack or hitting the recursion limit is a fact about the
    input's depth, not a defect."""
    v = analyzer.classify(output)
    assert v["is_bug"] is False and v["kind"] == "resource"


def test_ordinary_diagnostics_are_not_bugs():
    assert analyzer.classify(DIAGNOSTIC)["is_bug"] is False


def test_signature_is_stable_across_seeds():
    other = (DEBUG_ASSERT.replace("`Bar`", "`Quux`")
                         .replace("0: rust_begin_unwind", "0: rust_begin_unwind"))
    assert analyzer.crash_signature(DEBUG_ASSERT) == analyzer.crash_signature(other)


# ---------------------------------------------------------------------------
# Seed analysis — what the driver keys on
# ---------------------------------------------------------------------------

UNSAFE_SEED = '''//@ compile-flags: -Zmir-opt-level=4 --target=nonexistent -Lsomepath
#![feature(core_intrinsics)]

use std::mem::MaybeUninit;

fn main() {
    let p: *const u8 = std::ptr::null();
    unsafe {
        let x: u32 = std::mem::transmute(0u32);
        let _ = MaybeUninit::<u8>::uninit().assume_init();
        let _ = *p;
        let _ = x;
    }
}
'''

SAFE_SEED = "fn main() { let x = 1; println!(\"{}\", x); }\n"


def test_unsafe_score_separates_unsafe_seeds_from_safe_ones():
    """The driver spends sanitizer time only on seeds that use unsafe —
    instrumenting a seed with no raw pointers costs compile time and finds
    nothing."""
    assert analyzer.analyze_seed(UNSAFE_SEED)["unsafe_score"] > 3
    assert analyzer.analyze_seed(SAFE_SEED)["unsafe_score"] == 0


def test_compile_flags_are_honoured_but_path_and_target_flags_dropped():
    """A mir-opt test compiled without its `-Zmir-opt-level` exercises
    nothing. But `--target` and `-L` would fail in the driver rather than
    in the compiler, and the driver picks the target itself."""
    facts = analyzer.analyze_seed(UNSAFE_SEED)
    assert facts["compile_flags"] == ["-Zmir-opt-level=4"]


def test_features_and_crate_attrs_are_recorded():
    facts = analyzer.analyze_seed(UNSAFE_SEED)
    assert facts["features"] == ["core_intrinsics"]
    assert any("feature" in a for a in facts["crate_attrs"])
    assert facts["has_main"] is True


def test_known_bug_seeds_are_flagged():
    """`//@ known-bug` marks a test that is supposed to ICE. Reporting a hit
    on one is rediscovering a filed bug."""
    assert analyzer.analyze_seed("//@ known-bug: #123\nfn main() {}\n")["is_known_bug"]
    assert not analyzer.analyze_seed(SAFE_SEED)["is_known_bug"]


# ---------------------------------------------------------------------------
# Crate structure
# ---------------------------------------------------------------------------

SEED_A = '''#![feature(never_type)]
use std::fmt;

struct A(u32);

fn main() {
    let a = A(1);
    let _ = a.0;
}
'''

SEED_B = '''#![feature(box_patterns)]
use std::collections::HashMap;

struct B(String);

fn main() {
    let b = B(String::new());
    let _ = b.0;
}
'''


@pytest.mark.parametrize("technique,flags", [
    ("dataflow", {"dataflow_fusion": True}),
    ("state", {"state_fusion": True}),
    ("declaration", {"declaration_fusion": True}),
])
def test_inner_attributes_stay_at_the_crate_root(technique, flags):
    """`#![...]` is only legal at the top of the crate. Both seeds carry a
    feature gate, so a fusion that does not hoist them produces "an inner
    attribute is not permitted in this context" — rejected in the parser,
    before any compiler path worth testing."""
    strategies = get_strategies("rust", pre_analysis_enabled=True, **flags)
    if not strategies:
        pytest.skip(f"rust has no {technique} strategy")
    a = Seed(content=SEED_A, metadata={"filename": "a"})
    b = Seed(content=SEED_B, metadata={"filename": "b"})
    random.seed(11)
    for _ in range(20):
        child = strategies[0].fuse(a, b)
        lines = [ln for ln in child.content.splitlines() if ln.strip()]
        attr_positions = [i for i, ln in enumerate(lines)
                          if ln.strip().startswith("#![")]
        if not attr_positions:
            continue
        # Every inner attribute must precede every item.
        first_item = next((i for i, ln in enumerate(lines)
                           if ln.strip().startswith(("fn ", "struct ", "impl ",
                                                     "const ", "static "))),
                          len(lines))
        assert max(attr_positions) < first_item, child.content


def test_state_fusion_emits_a_main():
    """_process_seed renames each side's `main` apart so the splice cannot
    produce two of them — which leaves the crate with none, and the driver
    builds and runs some children."""
    strategy = get_strategies("rust", state_fusion=True,
                              pre_analysis_enabled=True)[0]
    a = Seed(content=SEED_A, metadata={"filename": "a"})
    b = Seed(content=SEED_B, metadata={"filename": "b"})
    random.seed(12)
    for _ in range(10):
        child = strategy.fuse(a, b)
        assert "fn main" in child.content


def test_rust_now_has_all_three_techniques():
    """Rust had no state fusion before this adapter was rebuilt; the
    registry entry was `None` even though LIVE_VAR_CONFIGS already had a
    brace-mode entry for it."""
    names = [type(s).__name__ for s in get_strategies("rust", pre_analysis_enabled=True)]
    assert names == ["RustFusionStrategy", "RustStructFusionStrategy",
                     "RustStateFusionStrategy"]


def test_config_declares_what_the_framework_reads():
    import re
    import yaml
    with open(os.path.join(ROOT, "projects", "rust", "config.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert cfg["project_name"] == "rust"
    patterns = cfg["analysis"]["crash_patterns"]
    assert "internal compiler error" in patterns
    # Live only because setup.py enables llvm.assertions.
    assert "LLVM ERROR" in patterns
    for entry in cfg["paths"]["seed_exclude_patterns"]:
        re.compile(entry["pattern"])
        assert entry["reason"]


def test_parser_emits_both_the_new_keys_and_the_ones_tests_pin():
    parser = _load("ffl_rust_parser_test", "projects/rust/parser.py")
    meta = parser._parser.parse_content(UNSAFE_SEED, "u.rs")
    assert meta["type"] == "rust"
    assert meta["variables"] and meta["dataflows"]
    assert meta["unsafe_score"] > 3
    assert meta["features"] == ["core_intrinsics"]
    # Unread by any strategy, but pinned by tests/projects/test_parsers.py.
    for key in ("imports", "functions", "structs"):
        assert key in meta


# ---------------------------------------------------------------------------
# De-collision must rename every occurrence, not a random subset
# ---------------------------------------------------------------------------

def test_decollision_renames_every_occurrence():
    """`replace_word_occurrences` rewrites a weighted-random *subset* — the
    right behaviour for dataflow fusion, where a partial rename is the
    connection being made, and the wrong one for de-collision.

    Renaming a declaration but only some of its uses turns one "defined
    multiple times" error into a pile of "cannot find type X in this
    scope". On the rust-lang/rust corpus that was 162 such errors per 120
    fused files, with the original collisions still half-unfixed.
    """
    from core.fusion import rename_all_word_occurrences
    src = "struct Foo;\nfn f(x: Foo) -> Foo { Foo }\nstruct Foobar;\n"
    out = rename_all_word_occurrences(src, "Foo", "Foo_b1")
    assert out.count("Foo_b1") == 4
    assert "Foo;" not in out.replace("Foo_b1;", "")
    # Word-boundary safety still holds: Foobar must not become Foo_b1bar.
    assert "Foobar" in out


def test_toplevel_collisions_are_fully_resolved():
    """Two seeds from tests/ui very often both declare `struct Foo`; Rust
    rejects the redeclaration outright, and it was about half of all
    rejections before de-collision existed."""
    strategy = get_strategies("rust", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    a_body = "struct Foo;\ntrait Bar {}\nfn helper() {}\n"
    b_body = "struct Foo;\ntrait Bar {}\nfn helper() -> Foo { Foo }\n"
    out, collisions = strategy._dedupe_toplevel(a_body, b_body, "b1")
    assert collisions == {"Foo", "Bar", "helper"}
    # Every reference moved with its declaration.
    assert "Foo_b1" in out and "Bar_b1" in out and "helper_b1" in out
    assert not [ln for ln in out.splitlines()
                if ln.strip() in ("struct Foo;", "trait Bar {}")]


def test_inner_doc_comments_are_hoisted_like_inner_attributes():
    """`//!` is crate-level in exactly the way `#![...]` is — legal only
    before any item. Leaving one mid-body gives "expected outer doc
    comment", which was the second-largest rejection class."""
    strategy = get_strategies("rust", dataflow_fusion=True,
                              pre_analysis_enabled=True)[0]
    attrs, _uses, body, _main = strategy._process_seed(
        "//! crate docs\n#![feature(x)]\nfn main() {}\n", "s1")
    assert any(a.strip().startswith("//!") for a in attrs)
    assert "//!" not in body
