---
name: fusion-fuzz-bug-triage
description: Triage a crash found by fusion-fuzz — decide whether it is a real target bug or an artefact of our own harness, prove it is fusion-specific, minimize it, generate a submittable report, and check upstream for duplicates. Use when asked to triage, investigate, reduce, or report a finding under output/bugs/.
---

# Fusion-Fuzz bug triage

A finding under `output/bugs/` is a **candidate**, not a bug. This document
is the procedure for turning one into either a filed report or a documented
dismissal.

## The rule that matters most

**Assume the finding is our own defect until you have proved otherwise.**

This is not caution for its own sake. On the V8 campaign, every one of the
first 20 findings was a defect in the adapter, not in V8:

| What was reported | What it actually was |
|---|---|
| `FATAL: isolate.cc:2522` | We passed `--correctness-fuzzer-suppressions`, which turns an ordinary JS stack overflow into a fatal error |
| `DCHECK: trigger-failure-extension.cc:49` | A corpus seed calling `triggerAssertFalse()`, whose body is literally `DCHECK(false)` |
| `DCHECK: bits.h:236`, `stack-guard.cc:250` | Our mutator rewrote integers inside the `// Flags:` **comment**, turning `--stack-size=100` into `--stack-size=2147483647` |
| 14 × `CORRECTNESS: ...` | We compared whole stdout between two configurations; the differences were exception text and stack-overflow line counts, which legitimately vary |

On CPython, 6 of 7 findings were likewise ours. **A finding that reproduces
is not yet a bug — it must also be the target's fault.**

The single cheapest test, which would have caught four of the five rows
above:

```bash
# Does it still "reproduce" with an EMPTY program and the same flags?
: > /tmp/empty.<ext>
<same command as test.sh, but on /tmp/empty.<ext>>
```

If an empty input reproduces it, the flags are the bug — not the program.

---

## Where bugs live and what is in a bundle

`output/bugs/<project>/<signature>/`, written by
`core/orchestrator.py::_save_crash_bundle`:

| File | What it is |
|---|---|
| `README.md` | Auto-generated report: signature, program, output, repro command, parents |
| `test.<ext>` | The fused reproducer |
| `min.<ext>` | **A copy of `test.<ext>`, not yet minimized.** It only becomes minimal after you run the reducer (step 4) |
| `test.out` | Combined stdout + stderr of the crashing run |
| `test.sh` | The exact reproducing command, with env prefix and flags |
| `parent_a.<ext>`, `parent_b.<ext>` | The two seeds that were fused |
| `report.md` | **You create this** (step 5) — the report to submit upstream |

Two traps in the file names:

- **CPython uses `ffl_repro.py`, not `test.py`.** `test` is a stdlib package,
  so a file named `test.py` shadows it and `import test.support` — which
  most of CPython's corpus does — fails with `'test' is not a package`
  instead of reproducing.
- `parent_a`/`parent_b` may be absent for a bug found by replay rather than
  fusion. Without them, step 3 cannot run; say so rather than guessing.

---

## Procedure

Work inside the project's container. Names are **not** consistent — look
them up, don't assume:

```bash
docker ps --format '{{.Names}}\t{{.Image}}'
```

At the time of writing: `fuzz-clang`, `fuzz-gcc`, `fuzz-cpython`,
`fuzz-rust`, `fuzz-go`, `fusion-v8`, `fusion-sm`.

### Step 1 — Read the output before touching anything

```bash
cd output/bugs/<project>/<signature>/
cat test.out | head -40      # the actual failure
cat test.sh                  # the flags it took
```

Ask, in this order:

1. **Is the failure in the target, or in the harness/shell?** A crash whose
   source file is the shell's own argument parsing (`shell/js.cpp`,
   `d8.cc`) is a bad command line, not an engine bug.
2. **Did we ask for this crash?** Check `test.<ext>` for deliberate-abort
   primitives — `triggerAssertFalse()`, `%AbortJS`, `%SystemBreak`,
   `crash()`, `oomAtAllocation()`. These exist to abort on demand.
3. **Is it resource exhaustion wearing a crash costume?** Engines route OOM
   through the same channel as real assertions: V8's `# Fatal error`, and
   SpiderMonkey's `MOZ_CRASH`. Look for `out of memory`, `Reached heap
   limit`, `too much recursion`, `hard rss limit exhausted`.
4. **Does the corpus declare this failure as correct?** SpiderMonkey
   jit-tests carry `// |jit-test| allow-oom; error: TypeError;
   exitstatus: 3`. A test that is *supposed* to throw is not a finding
   when it throws.
5. **Are any flags absurd?** Compare the values in `test.sh` against what
   the seed's own directive line says. A mutated `--stack-size=2147483647`
   is our defect.

If any of these fire, **stop**. Record the dismissal in `README.md` under
`## Triage` and fix the adapter so the class cannot recur — a dismissed
finding that stays dismissible will be re-found tomorrow.

### Step 2 — Reproduce standalone, then with an empty file

The reducers read from `/tmp`, so move the reproducer there:

```bash
docker exec -it <container> bash
cd /home/fuzz/WorkSpace/fusion-fuzz/output/bugs/<project>/<signature>/
cp test.<ext> /tmp/            # CPython: cp ffl_repro.py /tmp/
bash test.sh                   # confirm it still reproduces
```

Then the empty-file control from the top of this document. If the empty
file reproduces it, the finding is about the flags; go back to step 1.

Run it **three times**. A crash that reproduces once is nondeterminism.

### Step 3 — Is it fusion-specific?

The point of fusion-fuzz is bugs that a single seed does not find. A bug
that either parent triggers alone was already reachable without us.

Use the existing tool rather than re-implementing it — it re-runs the exact
same command against each parent and compares signatures:

```bash
python3 -m core.reproduce_check --project <project>
```

It reports each bug as *fusion-specific* or *reproducible from a parent
alone*, and writes a summary. For a single bug, do it by hand:

```bash
cp parent_a.<ext> /tmp/ && sed 's/test\.<ext>/parent_a.<ext>/' test.sh | bash
cp parent_b.<ext> /tmp/ && sed 's/test\.<ext>/parent_b.<ext>/' test.sh | bash
```

**Record the answer in `README.md` either way.** "Reproducible from
parent_a alone" is a legitimate and useful outcome — it is still possibly a
real bug, just not a fusion discovery, and the report should not claim
otherwise.

### Step 4 — Minimize and generate the report

`projects/<lang>/reduce.py` (note: `reduce.py`, not `reducer.py`) does
delta debugging on the program **and** on the flag set, then prints a
submittable report.

Every reducer is driven by constants at the bottom of the file, under
`if __name__ == "__main__":`. **Edit these before running** — copy the
values out of `test.sh`:

| Constant | Where it comes from |
|---|---|
| `testpath` | `/tmp/<the file you copied in step 2>` |
| binary path (`pypath`/`d8path`/`clang_bin`/…) | the executable named in `test.sh` |
| `config` (a string) or `flags` (a list) | the tokens in `test.sh` between the binary and the source file |
| `env_prefix` | the `ulimit`/`ASAN_OPTIONS` prefix in `test.sh`, if any |
| `bug_output` | a short, distinctive string from `test.out` |

`bug_output` is the one to get right. It is the oracle for every one of the
hundreds of runs the reducer makes, so:

- Pick something **specific to this crash** — the assertion expression, or
  `Debug check failed: <expr>`. Not `Error`, which matches ordinary output.
- Do **not** pick a string containing an address or a path; it will not
  match on the next run.

The flag set is reduced too, and this is where reducers most often "fail"
misleadingly: if the crash needs a flag you forgot to list, the reducer
strips flags until nothing reproduces and reports "bug not reproduced".
When that happens, re-check that `config`/`flags` match `test.sh` exactly.

Run it and save the report — the reducers print with ANSI colour, so strip
it:

```bash
python3 projects/<lang>/reduce.py | sed 's/\x1b\[[0-9;]*m//g' > report.md
```

Then copy the minimized program the reducer left in `/tmp` back over
`min.<ext>`, which until now was only a copy of `test.<ext>`:

```bash
cp /tmp/<testfile> min.<ext>
```

**Reducer coverage is not complete.** `projects/gcc/reduce.py` minimizes
but has no report template, and `go`, `spidermonkey` and `mlir` have no
`reduce.py` at all. For those, minimize by hand (bisect the file, keep the
half that still crashes) and write `report.md` from the template below.

### Step 5 — Check for duplicates before writing anything up

Search the upstream tracker for the **innermost meaningful stack frame** or
the **assertion expression** — not the whole signature, which contains our
paths and addresses.

| Project | Tracker |
|---|---|
| clang, flang, LLVM | https://github.com/llvm/llvm-project/issues |
| gcc | https://gcc.gnu.org/bugzilla/ |
| v8 | https://issues.chromium.org/ (component: Blink>JavaScript, or V8) |
| spidermonkey | https://bugzilla.mozilla.org/ (Core → JavaScript Engine / JavaScript Engine: JIT) |
| cpython | https://github.com/python/cpython/issues |
| rust | https://github.com/rust-lang/rust/issues |
| go | https://github.com/golang/go/issues |
| php | https://github.com/php/php-src/issues |
| swift | https://github.com/swiftlang/swift/issues |
| haskell (GHC) | https://gitlab.haskell.org/ghc/ghc/-/issues |
| naga | https://github.com/gfx-rs/wgpu/issues |

Search **closed issues too**: a bug fixed upstream but not yet in our
checkout is a duplicate, and the fix commit tells you so immediately.

Also check whether our own checkout is simply old:

```bash
cd projects/<lang>/<source-dir> && git log -1 --format='%H %cd'
```

Record what you searched and what you found in `README.md`. "Searched
`getPackAsArray` + `getKind() == Pack`, no open or closed match" is a
useful record; silence is not.

### Step 6 — Write it up

Append a `## Triage` section to `README.md` (do not delete the
auto-generated part — it carries the parent IDs and the original output):

```markdown
## Triage

- **Verdict:** real target bug | our defect | duplicate | not fusion-specific
- **Fusion-specific:** yes / no — parent_a: <reproduces?>, parent_b: <reproduces?>
- **Reproduced:** 3/3 runs, standalone in <container>
- **Empty-input control:** does not reproduce  ← rules out a flags-only artefact
- **Minimized:** <N> lines (from <M>), flags reduced to `<flags>`
- **Upstream search:** <terms searched, trackers, result>
- **Toolchain commit:** <hash> (<date>)
- **Notes:** <anything that would change the verdict>
```

`report.md` is what gets submitted, so it holds only what an upstream
maintainer needs — no fusion-fuzz internals, no parent IDs:

```markdown
The following code:

```<lang>
<minimized program>
```

Resulted in this output:

```
<the crash, trimmed to the assertion + the top ~15 frames>
```

To reproduce:

```
<binary> <reduced flags> ./min.<ext>
```

Commit:
```
<toolchain commit hash>
```

Build configuration:
```
<the configure/gn/cmake line from projects/<lang>/setup.py>
```

Operating System:
```
<distro, and the Docker image>
```
```

---

## Known false-positive classes

Check a new finding against this list before believing it. Each one cost a
real investigation.

| Class | How it looks | How to tell |
|---|---|---|
| Flag-manufactured crash | Crash in argument parsing, or an absurd flag value | Reproduces with an empty input file |
| Deliberate-abort primitive | `DCHECK(false)`, `abort:`, `MOZ_CRASH` from a `crash()` call | Grep the reproducer for the primitive |
| Resource exhaustion | OOM / recursion, reported through the assertion channel | `out of memory`, `too much recursion`, `Reached heap limit` |
| Corpus-declared failure | SpiderMonkey `// |jit-test| error:` / `allow-oom` | Read the seed's directive line |
| Sanitizer masking the assertion | ASan `SEGV ... Assertions.h:NNN in MOZ_CrashSequence` — identical for *every* assertion | Look for the `Assertion failure:` line above it; that is the real location |
| Our harness | Identifiers like `__ffl_`, `FFLDIGEST`, `__FFL_NAMES` in the reproducer | Re-run with `FFL_V8_HEAT=0` / `FFL_SM_HEAT=0` |
| Stale toolchain | Real bug, already fixed upstream | Check the checkout's commit date against the tracker |

## Judgement

Two failure modes are worth naming, because they pull in opposite
directions and both are costly:

- **Filing noise.** An upstream maintainer who receives one artefact stops
  reading the next report. Every dismissal above is cheaper than one bad
  filing.
- **Dismissing a real bug because it looked like the last false positive.**
  The checks here are concrete for exactly this reason: run the empty-input
  control, run the parent check, read the source of the assertion. Do not
  dismiss on resemblance.

When a check cannot be run — no parents saved, no reducer for the language,
the container is gone — say so in `README.md` and state what remains
unverified. An honest partial triage is useful; a confident one built on a
check you skipped is not.
