"""
Regression tests for the Fortran (flang/lfortran) fusion strategies.

Fortran has no braces: a source file is a flat sequence of program units,
at most one of which may be the main program, and a MODULE has to precede
whatever USEs it. Plain `code_a + "\\n" + code_b` concatenation breaks both
rules on nearly every pair, and a line-indexed state splice lands inside a
specification part more often than not. Measured against flang 22 on the
project corpus, that put the fused-child valid rate at 4.4%.

These tests pin the properties that fixed it:

  * a fused file declares exactly one main program — the donor's is
    demoted to a subroutine, whether it was `PROGRAM p ... END` or a
    header-less specification+execution part
  * chained fusion doesn't reuse the demoted name (it would redeclare it)
  * modules are hoisted ahead of the units that may USE them
  * state fusion merges the two specification parts in Fortran's mandated
    order (USE, then IMPORT, then a single IMPLICIT) and splices only in
    the execution part
  * dataflow fusion carries the donated name's declaration across, since
    the two halves stay in separate scopes
"""

import os
import random
import re
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.fusion import Seed, get_strategies  # noqa: E402

MAIN_A = """program alpha
  implicit none
  integer :: i
  i = 1
  print *, i
end program alpha"""

MAIN_B = """program beta
  use iso_fortran_env
  implicit none
  real :: x
  x = 2.0
  print *, x
end program beta"""

BARE_MAIN = """integer :: k
k = 7
print *, k
end"""

WITH_MODULE = """module helper
  implicit none
  integer, parameter :: unit_size = 4
end module helper

program gamma
  use helper
  implicit none
  integer :: n
  n = unit_size
end program gamma"""


def _seed(content, ident):
    return Seed(content=content, metadata={"type": "fortran", "extension": ".f90",
                                           "filename": ident})


def _strategies():
    return get_strategies("flang", dataflow_fusion=True, state_fusion=True,
                          declaration_fusion=True, pre_analysis_enabled=False)


def _by_name(name):
    for strategy in _strategies():
        if type(strategy).__name__ == name:
            return strategy
    pytest.skip(f"{name} unavailable")


def _count_main_programs(code):
    """PROGRAM headers plus header-less main programs, as flang counts them."""
    strategy = _by_name("FlangFusionStrategy")
    units, ok = strategy._split_top_level(code)
    assert ok, f"unit split did not balance:\n{code}"
    return sum(1 for kind, _ in units if kind == "program")


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("donor", [MAIN_B, BARE_MAIN])
def test_dataflow_fusion_keeps_one_main_program(seed, donor):
    random.seed(seed)
    strategy = _by_name("FlangFusionStrategy")
    for child in strategy.fuse_bidirectional(_seed(MAIN_A, "a"), _seed(donor, "b")):
        assert _count_main_programs(child.content) == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5, 6, 7, 8])
def test_state_fusion_keeps_one_main_program(seed):
    random.seed(seed)
    strategy = _by_name("FlangStateFusionStrategy")
    for child in strategy.fuse_bidirectional(_seed(MAIN_A, "a"), _seed(MAIN_B, "b")):
        assert _count_main_programs(child.content) == 1


def test_chained_fusion_does_not_reuse_the_demoted_name():
    """The orchestrator feeds a fused child back in as the next host, so a
    fixed donor-main name would be declared twice in the same file."""
    random.seed(11)
    strategy = _by_name("FlangFusionStrategy")
    intermediate = strategy.fuse(_seed(MAIN_A, "a"), _seed(MAIN_B, "b"))
    child = strategy.fuse(intermediate, _seed(BARE_MAIN, "c"))
    names = re.findall(r'(?im)^\s*subroutine\s+(ffl_donor_main\w*)', child.content)
    assert len(names) == len(set(names)) == 2


def test_modules_are_hoisted_ahead_of_their_users():
    random.seed(3)
    strategy = _by_name("FlangFusionStrategy")
    child = strategy.fuse(_seed(MAIN_A, "a"), _seed(WITH_MODULE, "b"))
    module_at = child.content.lower().index("module helper")
    use_at = child.content.lower().index("use helper")
    assert module_at < use_at


def test_state_fusion_merges_specification_parts_in_order():
    """USE before IMPLICIT, and only one IMPLICIT in the merged scope."""
    random.seed(5)
    strategy = _by_name("FlangStateFusionStrategy")
    child = strategy.fuse(_seed(MAIN_A, "a"), _seed(MAIN_B, "b"))
    lines = [line.strip().lower() for line in child.content.splitlines()]
    body = lines[:lines.index("end program alpha")] if "end program alpha" in lines else lines
    implicits = [i for i, line in enumerate(body) if line.startswith("implicit")]
    uses = [i for i, line in enumerate(body) if line.startswith("use ")]
    assert len(implicits) == 1
    assert all(u < implicits[0] for u in uses)


def test_dataflow_fusion_declares_the_name_it_donates():
    """A and B stay separate program units, so a bare name from A spliced
    into B is undeclared there unless its declaration comes along."""
    random.seed(2)
    strategy = _by_name("FlangFusionStrategy")
    donor = """program beta
  implicit none
  real :: x
  x = 2.0
  print *, x
end program beta"""
    host = """program alpha
  implicit none
  integer :: theta
  theta = 1
end program alpha"""
    child = strategy._build_fused_test(_seed(host, "a"), _seed(donor, "b"), "df_ab")
    lowered = child.content.lower()
    if re.search(r'(?<!\w)theta(?!\w)', lowered.split("subroutine")[-1]):
        assert re.search(r'::\s*theta', lowered)


def test_unit_split_survives_select_type_and_labelled_do():
    """`TYPE IS (...)` and `DO 10 I=...` are statements, not block openers;
    counting them as openers desynchronises every depth-based scan."""
    code = """program tricky
  implicit none
  class(*), allocatable :: v
  integer :: i
  allocate(integer :: v)
  select type (v)
  type is (integer)
    v = 1
  class default
    continue
  end select
  do 10 i = 1, 3
10 continue
end program tricky"""
    strategy = _by_name("FlangFusionStrategy")
    units, ok = strategy._split_top_level(code)
    assert ok
    assert [kind for kind, _ in units] == ["program"]
