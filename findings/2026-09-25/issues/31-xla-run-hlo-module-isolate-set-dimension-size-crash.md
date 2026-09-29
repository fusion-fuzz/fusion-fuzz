# [XLA] `run_hlo_module --isolate_instructions` crashes on a module with `set-dimension-size`

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

`--isolate_instructions=true` runs every instruction of the entry as its own module. For a
`set-dimension-size` the isolated module's result is a dynamic literal, and both runners
fall over:

* CPU: segmentation fault (`absl::AnyInvocable` invoked with a null `invoker_`,
  `internal/any_invocable.h:766` when assertions are on);
* interpreter: abort in `Shape::CheckDimensionSize` from `LiteralBase::ToStatic`
  (`InterpreterLoadedExecutable::ExecuteSharded`).

Without the flag the module runs on both platforms and the CPU and interpreter results
agree.

## Reproducer

5 lines; the size (3) is inside the bound (8), so nothing about the module is invalid:

```
ENTRY main {
  param = s32[8] parameter(0)
  size = s32[] constant(3)
  param_dynamic = s32[<=8] set-dimension-size(param, size), dimensions={0}
}
```

```bash
run_hlo_module --platform=cpu --random_init_input_literals=true --reference_platform= \
  --isolate_instructions=true sds.hlo            # SIGSEGV
run_hlo_module --platform=interpreter --random_init_input_literals=true --reference_platform= \
  --isolate_instructions=true sds.hlo            # abort
```

## Expected

Isolation should either produce a module whose result the runners can materialise (a
static literal via `ToStatic` guarded by a status), or skip instructions with dynamic
result shapes with a message, rather than crash.
