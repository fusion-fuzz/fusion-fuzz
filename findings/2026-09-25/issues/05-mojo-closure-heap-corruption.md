# [Mojo] Heap corruption compiling nested closures with capture clauses (`double free or corruption`, SIGSEGV)

**Repo:** modular/modular · **Commit:** `0103072` (Mojo 1.2.0.dev2026092205), compiler built from source with assertions (`-c opt --copt=-UNDEBUG`)

## What happens

The compiler corrupts its own heap, so the symptom moves between runs of the same input.
Five runs of the reproducer below:

| Run | Exit | Message |
|---|---|---|
| 1 | 134 | `double free or corruption (!prev)` |
| 2 | 134 | crash banner only |
| 3 | 134 | crash banner only |
| 4 | 134 | `double free or corruption (!prev)`, `malloc(): invalid size (unsorted)` |
| 5 | 139 | SIGSEGV |

Other runs of the unreduced input produced `corrupted size vs. prev_size while
consolidating` and `free(): double free detected in tcache 2`.

## Reproducer

`closures.mojo`:

```mojo
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
```

```bash
mojo build --emit object -O3 --debug-level none -DASSERT=none -j 1 \
  -I Mojo/stdlib/test closures.mojo
```

## Expected

Compile the file or report an error. The input mixes nested `def` closures declared inside
`for` loops with `{imm}` / `{mut}` capture clauses and `async def` closures capturing a
`String`; several declarations are questionable, but no input should corrupt the
compiler's heap.

## Notes

Reduced from 128 to 29 lines. A run that exits 134 with only the crash banner is the same
defect. Building the compiler under AddressSanitizer would give a precise site.
