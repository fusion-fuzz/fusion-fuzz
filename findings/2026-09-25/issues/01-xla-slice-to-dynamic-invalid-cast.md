# [XLA:GPU] `SliceToDynamic` with a non-integer size operand asserts in LLVM: `Invalid cast!`

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG` (assertions on), `--config=cuda_nvcc`

## What happens

A module whose `SliceToDynamic` custom call receives a floating-point tensor where an
`s32` scalar size is expected passes the HLO verifier, and the GPU backend then aborts
inside LLVM while emitting code:

```
hlo-opt: llvm/lib/IR/Instructions.cpp:3111:
  static CastInst *llvm::CastInst::Create(Instruction::CastOps, Value *, Type *,
  const Twine &, InsertPosition): Assertion `castIsValid(op, S, Ty) && "Invalid cast!"' failed.
```

## Reproducer

`slice_to_dynamic.hlo`:

```
HloModule m

ENTRY %e {
  %scale = f8e8m0fnu[3,128,4] parameter(0)
  %p0 = f32[<=512] parameter(1)
  %c = (f32[512], s32[]) custom-call(%p0), custom_call_target="PadToStatic"
  %gte0 = f32[512] get-tuple-element(%c), index=0
  ROOT %c2 = f32[<=1024] custom-call(%gte0, %scale), custom_call_target="SliceToDynamic"
}
```

```bash
hlo-opt --platform=gpu --stage=llvm \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb \
  --xla_gpu_autotune_level=0 --o=/dev/null slice_to_dynamic.hlo
```

`--stage=ptx` and `--stage=buffer-assignment` abort in the same place. No GPU is needed:
the device description comes from the target-config proto. `--platform=cpu --stage=hlo`
(HloVerifier) accepts the module, which is why it reaches codegen.

## Expected

An error naming the malformed custom call. `SliceToDynamic`'s size operands are `s32[]`
scalars; an `f8e8m0fnu[3,128,4]` operand should be rejected by the verifier or by the
emitter rather than converted with an invalid LLVM cast.

## Notes

Custom-call operand types are not verified, so this is a robustness gap rather than a
miscompilation. Found by fusing two of XLA's own test modules and compiling for the GPU
backend on a host without a GPU.
