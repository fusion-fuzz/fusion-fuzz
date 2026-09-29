# [XLA] HLO parser CHECK-fails on a computation whose parameter numbers are not 0..n-1

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`

## What happens

The parser diagnoses the problem and then aborts instead of returning the error:

```
F0000 hlo_computation.cc:185] Check failed: param_no >= 0 && param_no < parameter_count
ERROR: invalid parameter number. Expected [0, 1), got 1
```

Stack: `HloComputation::HloComputation` ← `HloComputation::Builder::Build` ←
`HloParserImpl::ParseInstructionList` ← `HloParserImpl::Run` ←
`ParseAndReturnUnverifiedModule`. `--emit-proto` (parse only) is enough to hit it, so no
pass or backend is involved.

## Reproducer

Three lines:

```
%c {
  %p = s32[] parameter(1)
}
```

```bash
hlo-opt --platform=cpu --emit-proto --o=/dev/null param.hlo
```

`parameter(0)` parses; `parameter(1)`, `parameter(2)`, `parameter(3)` all abort — any
computation whose parameter numbers are not exactly `0..n-1`.

## Expected

A parse error. `ParseAndReturnUnverifiedModule` returns a `Status` precisely so that
malformed text can be rejected; the CHECK in the `HloComputation` constructor turns this
input into a crash of every tool that loads HLO text (`hlo-opt`, `run_hlo_module`, ...).

## Notes

Fuzzing produces this shape whenever a fused module keeps one parent's parameter
numbering; the same abort presumably applies to hand-edited HLO dumps.
