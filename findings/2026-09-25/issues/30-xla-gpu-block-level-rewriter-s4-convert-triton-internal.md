# [XLA:GPU] The block-level fusion rewriter hands an s4 convert to Triton, which cannot compile it

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`, `--config=cuda_nvcc`

## What happens

With `--xla_gpu_experimental_enable_fusion_block_level_rewriter=true`, a plain
`convert(s4 -> s8)` is wrapped into a `kCustom` / `__triton` fusion with a block-level
config, and codegen then fails:

```
INTERNAL: Failed to compile Triton kernel. Context: [Fusion: wrapped_convert = s8[2,2]{1,0} fusion(x),
  kind=kCustom, calls=wrapped_convert_computation, backend_config={... "kind":"__triton",
  "block_level_fusion_config":{"num_warps":"1","output_tiles":[{"sizes":["1","1"]}], ...}}]
```

The rewriter's "can Triton handle this fusion" check accepts the sub-byte operand; the
Triton emitter does not. The actual Triton diagnostic is not surfaced — only the context.
Without the flag the module compiles.

## Reproducer

5 lines:

```
HloModule TupleOutput
ENTRY main {
  x = s4[2,2] parameter(0)
  y = s8[2,2] convert(x)
  ROOT t = (s4[2,2], s8[2,2]) tuple(x, y)
}
```

```bash
hlo-opt --platform=gpu --stage=ptx \
  --xla_gpu_target_config_filename=xla/backends/gpu/target_config/specs/h100_sxm.txtpb \
  --xla_gpu_experimental_enable_fusion_block_level_rewriter=true --o=/dev/null s4_convert.hlo
```

Same on GB300; no GPU is needed.

## Expected

The rewriter should leave fusions with sub-byte types to the legacy emitters (or the
Triton support check should reject them), and the Triton diagnostic should be attached
to the INTERNAL status.
