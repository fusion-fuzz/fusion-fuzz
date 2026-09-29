# [XLA:CPU] Executing an entry that returns the same token twice CHECK-fails in `cpu_client.cc` ("Unexpected duplicate")

**Repo:** openxla/xla · **Commit:** `a8f4eee` (main, 2026-09-21) · **Build:** `-c opt --copt=-UNDEBUG`

## What happens

```
F0000 cpu_client.cc:1162] Check failed: output_indices_[result_buffer_indices_[i]] == -1 (0 vs. -1) Unexpected duplicate.
```

The module compiles; the CPU client aborts while wiring the result buffers, because the
two elements of the output tuple share one buffer. A duplicated *array* value in the
output tuple (`tuple(%p, %p)`, or the same `add` twice) runs fine; a duplicated token
does not.

## Reproducer

Five lines, accepted by the HLO verifier:

```
HloModule m

ENTRY %e {
  %tok = token[] after-all()
  ROOT %t = (token[], token[]) tuple(%tok, %tok)
}
```

```bash
run_hlo_module --platform=cpu --random_init_input_literals=true --reference_platform= dup_token.hlo
```

The same abort with the token coming from an `outfeed` instead of `after-all`.

## Expected

Either run the module (the same token appearing twice in a result tuple is well-formed
HLO) or reject it before execution with a status, as is done for other unsupported result
shapes. A CHECK in the client is neither.
