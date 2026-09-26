from std.runtime._asyncrt import (
    create_raising_task,
    create_task,
)
from std.testing import assert_equal, TestSuite
def outer(K: Int):
        for ko in range(0, K, 16):
            def inner[tile_n: Int](n_idx: Int) {imm}:
                var x = ko + n_idx * tile_n
def plain_fn_loop_capture(K: Int):
    for g in range(0, K, 16):
        def pack(n: Int) {imm} -> Int:
            return g + n
    async def test_asyncrt_add[lhs: Int](rhs: Int) -> Int:
        return lhs + rhs
    async def return_value[value: Int]() -> Int:
        return value
    async def run_as_group() -> Int:
        var t0 = create_task(return_value[1]())
        var t1 = create_task(return_value[2]())
        return await t0 + await t1
def test_runtime_unified_async_memory_result_raises_d() raises:
    var prefix = String("hello")
    async def build_message() raises {mut prefix} -> String:
        return prefix + String(" world")
    var task = create_raising_task(build_message())
    var result = task^.wait()
def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
