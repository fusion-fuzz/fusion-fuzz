"""
For every crash in output/bugs/<project>/, re-run the exact same reproducing
command (same flags, same driver) against each saved parent_a/parent_b
program instead of the fused test. If a parent alone reproduces a crash with
the same signature, the bug isn't a fusion discovery — it already exists in
the original (un-fused) seed. Reports which bugs are fusion-specific (crash
only with the fused program, not either parent alone).

Must run where the actual compiler toolchain is available (inside the
fuzz-clang container).

Usage:
    python3 -m core.reproduce_check --project clang
"""
import argparse
import importlib.util
import inspect
import os
import re
import subprocess
import sys


def extract_frames(signature, max_frames=3):
    """Return up to `max_frames` fully-qualified stack frame names from a
    driver-built 'Stack dump: ...' signature. Skips unsymbolized
    'lib+0xoffset' fallback frames, which aren't useful GitHub search terms.

    Two signature shapes come out of _stack_dump_fingerprint:
      - 'Stack dump: <msg> [f1 > f2 > f3]'  (message + frames)
      - 'Stack dump: f1 > f2 > f3'          (no message captured — bare,
                                              un-bracketed frame chain)
    Frames are split on the literal ' > ' (with surrounding spaces), not a
    bare '>' — heavily-templated C++ symbols (std::optional<Expr<...>>,
    the norm for Fortran::evaluate::* frames) contain internal '>'
    characters from template syntax that a bare-'>' split would wrongly
    break apart."""
    if not signature or not signature.startswith("Stack dump:"):
        return []
    m = re.search(r'\[(.*)\]\s*$', signature)
    if m:
        frame_blob = m.group(1)
    else:
        rest = signature[len("Stack dump:"):].strip()
        # Only treat the bare remainder as a frame chain if it actually
        # looks like one (qualified C++ name or multiple ' > '-joined
        # frames) — guards against ever mistaking a bare crash-site
        # message ("current parser token 'x'") for frame content.
        frame_blob = rest if ('::' in rest or ' > ' in rest) else ''
    if not frame_blob:
        return []
    frames = [f.strip() for f in frame_blob.split(' > ')]
    frames = [f for f in frames if f and "+0x" not in f]
    return frames[:max_frames]


def signatures_match(sig_a, sig_b):
    """Whether two crash signatures represent the same underlying bug.

    For Stack dump crashes, the crash-site "message" (current parser token
    'X', <eof> parser at end of file, ...) is *positional* — it depends on
    exactly where in the file the crash happens to occur, which shifts
    between the fused program and a parent compiled alone even when it's
    the identical clang bug. Comparing full signature strings for equality
    misses this: e.g. a fused crash reporting "current parser token 'int'"
    and the same bug reproduced from parent_b alone reporting "<eof> parser
    at end of file" share the exact same 3-frame call chain
    (emitBuiltinOSLogFormat > EmitBuiltinExpr > EmitCallExpr) and are the
    same bug, but differ as strings. Compare by frame chain instead;
    fall back to exact string equality for non-stack-dump signatures
    (ASAN/UBSAN/LLVM ERROR/Assertion/bare Segfault/Aborted), which don't
    have this positional-message issue."""
    if not sig_a or not sig_b:
        return False
    if sig_a == sig_b:
        return True
    if sig_a.startswith("Stack dump:") and sig_b.startswith("Stack dump:"):
        frames_a, frames_b = extract_frames(sig_a), extract_frames(sig_b)
        return bool(frames_a) and frames_a == frames_b
    return False


def _load_driver(project_name):
    from core.driver import BaseDriver
    from core.config_loader import load_project_config

    config = load_project_config(project_name)
    driver_path = os.path.join("projects", project_name, "driver.py")
    spec = importlib.util.spec_from_file_location(f"ffl_{project_name}_driver", driver_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for _, obj in inspect.getmembers(mod):
        if inspect.isclass(obj) and issubclass(obj, BaseDriver) and obj is not BaseDriver:
            return obj(config)
    raise RuntimeError(f"No BaseDriver subclass found in {driver_path}")


def _extract_command(test_sh_text):
    """Return the last non-empty, non-comment, non-SCRIPT_DIR line — the
    actual reproducing command. The line still references "$SCRIPT_DIR"
    (defined by the line we just dropped), so resolve it to "." — callers
    always run with cwd=bug_dir, which is exactly what SCRIPT_DIR pointed
    to. Without this substitution $SCRIPT_DIR is unset in our subshell and
    silently expands to "", turning "$SCRIPT_DIR/test.c" into "/test.c" —
    an absolute path to a file that doesn't exist, so nothing reproduces."""
    lines = [ln for ln in test_sh_text.splitlines() if ln.strip()
              and not ln.strip().startswith('#') and 'SCRIPT_DIR=' not in ln]
    if not lines:
        return None
    return lines[-1].replace("$SCRIPT_DIR", ".").replace("${SCRIPT_DIR}", ".")


def _run(cmd, cwd, timeout):
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=cwd, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        return proc.returncode, proc.stdout.decode("utf-8", "replace"), proc.stderr.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as e:
        return 1, "", str(e)


def check_bug_dir(bug_dir, driver, timeout=10):
    """Returns a dict describing whether this bug reproduces from a parent."""
    test_sh_path = os.path.join(bug_dir, "test.sh")
    if not os.path.isfile(test_sh_path):
        return None
    cmd_template = _extract_command(open(test_sh_path, encoding="utf-8", errors="replace").read())
    if not cmd_template:
        return None

    # Figure out the fused test's extension from whichever test.<ext> exists.
    test_files = [f for f in os.listdir(bug_dir) if f.startswith("test.") and f != "test.sh" and f != "test.out"]
    if not test_files:
        return None
    ext = "." + test_files[0].split(".", 1)[1]

    fused_path_frag = f"test{ext}"

    # Re-verify the fused crash itself with the exact recorded command,
    # to get a canonical signature computed with the *current* driver logic.
    rc, out, err = _run(cmd_template, cwd=bug_dir, timeout=timeout)
    fused_crashed = driver._check_crash(out, err, rc)
    fused_sig = driver.extract_crash_signature(out, err, rc) if fused_crashed else None

    result = {
        "bug_dir": os.path.basename(bug_dir),
        "fused_reproduced": fused_crashed,
        "fused_signature": fused_sig,
        "parents": {},
    }

    for label in ("parent_a", "parent_b"):
        parent_path = os.path.join(bug_dir, f"{label}{ext}")
        if not os.path.isfile(parent_path):
            continue
        parent_cmd = cmd_template.replace(fused_path_frag, f"{label}{ext}")
        p_rc, p_out, p_err = _run(parent_cmd, cwd=bug_dir, timeout=timeout)
        p_crashed = driver._check_crash(p_out, p_err, p_rc)
        p_sig = driver.extract_crash_signature(p_out, p_err, p_rc) if p_crashed else None
        result["parents"][label] = {
            "crashed": p_crashed,
            "signature": p_sig,
            "same_signature_as_fused": bool(p_crashed and signatures_match(fused_sig, p_sig)),
        }

    return result


def run(project, bugs_dir=None, timeout=10):
    bugs_dir = bugs_dir or os.path.join("output", "bugs", project)
    driver = _load_driver(project)

    bug_dirs = sorted(
        d for d in os.listdir(bugs_dir)
        if os.path.isdir(os.path.join(bugs_dir, d))
    )

    fusion_specific = []
    parent_reproducible = []
    skipped = []

    for i, name in enumerate(bug_dirs, 1):
        path = os.path.join(bugs_dir, name)
        print(f"[{i}/{len(bug_dirs)}] {name}", flush=True)
        res = check_bug_dir(path, driver, timeout=timeout)
        if res is None:
            skipped.append(name)
            continue
        if not res["fused_reproduced"]:
            print("  -> fused test no longer reproduces (flaky/env-dependent) — skipping")
            skipped.append(name)
            continue

        hit = [label for label, info in res["parents"].items() if info["same_signature_as_fused"]]
        if hit:
            print(f"  -> ALSO reproduces in {', '.join(hit)} — not fusion-specific")
            parent_reproducible.append((name, res))
        else:
            print("  -> does NOT reproduce in either parent — fusion-specific")
            fusion_specific.append((name, res))

    print()
    print("=" * 70)
    print(f"Total bug folders examined: {len(bug_dirs)}")
    print(f"Skipped (no test.sh/parents, or no longer reproduces): {len(skipped)}")
    print(f"Reproducible from a parent alone (NOT fusion-specific): {len(parent_reproducible)}")
    print(f"Fusion-specific (crash requires the fused program): {len(fusion_specific)}")
    print("=" * 70)

    report_path = os.path.join(bugs_dir, "PARENT_REPRO_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Parent-reproducibility check — {project}\n\n")
        f.write(f"- Total examined: {len(bug_dirs)}\n")
        f.write(f"- Skipped: {len(skipped)}\n")
        f.write(f"- Reproducible from a parent alone (not fusion-specific): {len(parent_reproducible)}\n")
        f.write(f"- Fusion-specific: {len(fusion_specific)}\n\n")

        f.write("## Fusion-specific bugs (crash only with the fused program)\n\n")
        for name, res in fusion_specific:
            f.write(f"- `{name}` — signature: `{res['fused_signature']}`\n")
        f.write("\n")

        f.write("## Reproducible from a parent alone (filter out / not novel)\n\n")
        for name, res in parent_reproducible:
            hits = [label for label, info in res["parents"].items() if info["same_signature_as_fused"]]
            f.write(f"- `{name}` — also crashes via {', '.join(hits)} — signature: `{res['fused_signature']}`\n")
        f.write("\n")

        if skipped:
            f.write("## Skipped\n\n")
            for name in skipped:
                f.write(f"- `{name}`\n")

    print(f"\nReport written to {report_path}")
    return fusion_specific, parent_reproducible, skipped


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", required=True)
    ap.add_argument("--bugs-dir", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args()
    run(args.project, bugs_dir=args.bugs_dir, timeout=args.timeout)
