# [XLA:CPU] Buffer assignment CHECK-fails on an async-update inside a while loop

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 buffer_assignment.cc:2032] Check failed: it2 != buffer_live_ranges.end()
  Buffer doesn't have a proper live range:<1 custom-call @0>
```

`BufferAssigner::MaybeAssignBuffer` (via `TryAssignToExistingAllocation` /
`AssignSingleHloBuffer`) looks up the live range of the buffer defined by the custom-call
inside the async computation and finds none. The async-start is outside the loop, the
async-update is in the loop body and the async-done comes after the loop — the shape of
XLA's own `CopyInsertion` unit test for async ops that live across a while. A variant with
a second while loop consuming the async-done hits the sibling CHECK at line 2015
(`it != buffer_live_ranges.end()`).

## Reproducer

28 lines; the HLO verifier and the whole HLO pass pipeline accept it (`--stage=hlo` exits
0); the CHECK fires when buffer assignment runs.

```
async_computation {
  p = f32[1024] parameter(0)
  ROOT custom-call = f32[1024] custom-call(p), custom_call_target="foo"
}
while_cond {
  param = (s32[], ((f32[1024]), f32[1024], s32[])) parameter(0)
  count = s32[] get-tuple-element(param), index=0
  limit = s32[] constant(10)
  ROOT cmp = pred[] compare(count, limit), direction=LT
}
while_body {
  param = (s32[], ((f32[1024]), f32[1024], s32[])) parameter(0)
  count = s32[] get-tuple-element(param), index=0
  one = s32[] constant(1)
  new_count = s32[] add(count, one)
  async_state = ((f32[1024]), f32[1024], s32[]) get-tuple-element(param), index=1
  async_update = ((f32[1024]), f32[1024], s32[]) async-update(async_state), calls=async_computation
  ROOT body_root = (s32[], ((f32[1024]), f32[1024], s32[])) tuple(new_count, async_update)
}
ENTRY main {
  p0 = f32[1024] parameter(0)
  async-start = ((f32[1024]), f32[1024], s32[]) async-start(p0), calls=async_computation
  start_count = s32[] constant(0)
  iter_init = (s32[], ((f32[1024]), f32[1024], s32[])) tuple(start_count, async-start)
  while_loop = (s32[], ((f32[1024]), f32[1024], s32[])) while(iter_init), condition=while_cond, body=while_body
  final_async_state = ((f32[1024]), f32[1024], s32[]) get-tuple-element(while_loop), index=1
  ROOT async-done = f32[1024] async-done(final_async_state), calls=async_computation
}
```

```bash
hlo-opt --platform=cpu --stage=llvm-before-optimizations --o=/dev/null async_while.hlo
```

## Expected

Either copy insertion / the async-op passes should rewrite the module into something
buffer assignment handles, or the module should be rejected with a status. A CHECK deep in
buffer assignment on a verifier-accepted module is neither.
