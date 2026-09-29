# [XLA] `shardy-xla` asserts (`resultNumber < getNumResults()`) importing a while body whose root is its parameter next to an outfeed

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
hlo-opt: .../mlir/IR/Operation.h:1069: mlir::OpResult mlir::Operation::getResult(unsigned int):
  Assertion `resultNumber < getNumResults() && "Result number is out of range for op"' failed.
```

The Shardy round-trip (`hlo-opt --passes=shardy-xla`) imports the module to MLIR and
back. The while body's root is its own parameter (`ROOT arg.0 = parameter(0)`); the
body also contains an `outfeed` that is not on the path to the root. Something in the
import/export asks a zero-result op for a result.

## Reproducer

18 lines; the verifier accepts it and both the CPU and GPU pipelines compile it
(`--stage=hlo`), so this is only reachable when the Shardy pass is run on its own or on
a module with shardings:

```
call_body {
  ROOT param.0 = s32[] parameter(0)
}
body {
  ROOT arg.0 = s32[] parameter(0), sharding={replicated}
  token.0 = after-all()
  outfeed.0 = token[] outfeed(arg.0, token.0), outfeed_shape=s32[]
}
cond {
  arg.0 = s32[] parameter(0), sharding={replicated}
  ROOT true.0 = pred[] constant(true)
}
ENTRY main {
  arg.0 = s32[] parameter(0)
  call.0 = s32[] call(arg.0), to_apply=call_body
  call.1 = s32[] call(call.0), to_apply=call_body
  ROOT while.0 = s32[] while(call.1), condition=cond, body=body
}
```

```bash
hlo-opt --platform=cpu --passes=shardy-xla --o=/dev/null shardy_while_outfeed.hlo
```

## Expected

The Shardy import should either handle a body whose root is a parameter with a dangling
side-effecting op, or return a status; an MLIR assertion is neither.
