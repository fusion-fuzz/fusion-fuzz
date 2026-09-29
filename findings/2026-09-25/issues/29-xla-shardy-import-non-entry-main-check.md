# [XLA] Shardy import CHECK-fails when a non-entry computation is named `main`

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

```
F0000 module_attributes_importer.cc:351] Check failed: parameter_shapes.size() == main.getNumArguments() (2 vs. 1)
```

`ShardyXLA` imports the HLO module to MLIR and then looks up the function called `main`
to attach the entry computation's attributes. HLO allows any computation to be named
`main`; when a non-entry computation carries that name the importer picks it up and the
entry's parameter count no longer matches. Without any sharding in the module the same
name clash surfaces as a status instead (`conversion requires module with `main`
function`), which `hlo-opt` then CHECKs on in `opt_lib.cc:195`.

## Reproducer

12 lines, line-reduced; the verifier and the CPU pipeline accept it:

```
HloModule module
%recovery_2 (p: f32[2,16]) -> f32[4,16] {
  %p = f32[2,16]{1,0} parameter(0), sharding={devices=[2,1]<=[2]}
  ROOT %ag = f32[4,16]{1,0} all-gather(%p), dimensions={0}
}
%main (p: f32[4,8]) -> f32[4,8] {
  %p = f32[4,8]{1,0} parameter(0)
}
ENTRY %fusion {
  %param0 = s4[200]{0:E(4)} parameter(0)
  %param1 = s4[400]{0:E(4)} parameter(1)
}
```

```bash
hlo-opt --platform=cpu --passes=shardy-xla --o=/dev/null nonentry_main.hlo
```

Renaming `%main` to anything else makes the pass succeed.

## Expected

The importer should locate the entry computation by its role (the module's entry), not
by the string `main`, or uniquify the clashing name before import; a CHECK is not an
acceptable outcome for a legal module.
