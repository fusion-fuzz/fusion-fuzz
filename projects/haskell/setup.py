import concurrent.futures
import glob
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

_HARVEST_TIMEOUT = 4
_HARVEST_WORKERS = 16

# Separate caps for ticket-numbered vs. generically-named GHC testsuite
# candidates. should_run+should_compile+flat-categories together contribute
# ~2700+ ticket-numbered candidates — comfortably more than any single cap
# — so a single pooled cap would let ticketed tests crowd out every
# non-ticketed one (an entire source like th/*.hs among them) before it's
# ever even attempted. Capping the two buckets independently guarantees
# both get real representation.
_GHC_TICKET_CAP = 700
_GHC_OTHER_CAP = 500

# Rosetta Code and Exercism have no "ticket-numbered" concept — flat caps.
_ROSETTA_CAP = 400
_EXERCISM_CAP = 150

# nofib is small (515 .hs files total) and heavy on multi-module benchmarks
# (e.g. real/ is almost entirely multi-file programs) — cap is a ceiling,
# not a target; the viability filter is expected to keep well under it.
_NOFIB_CAP = 300

# CodeNet's Haskell subset is competitive-judge submissions (AtCoder/AIZU),
# so almost all are single-file by construction — cap generously.
_CODENET_CAP = 1000

# GHC testsuite files named T<gitlab-issue-number>(suffix).hs are literal
# regression tests for historic bug reports (e.g. T9254.hs pins down
# https://gitlab.haskell.org/ghc/ghc/-/issues/9254) — prioritize these over
# generically-named ones so the harvested corpus skews toward real bug
# history rather than arbitrary feature-coverage snippets.
_TICKET_NUMBERED_RE = re.compile(r'^T\d+[a-zA-Z_]*\.hs$')


def _resolve_ghc():
    found = shutil.which("ghc")
    if found:
        return found
    matches = glob.glob("/opt/ghc/*/bin/ghc")
    return sorted(matches)[-1] if matches else "ghc"


def _run_with_group_kill(cmd, cwd, timeout):
    """Run cmd, killing the whole process group on timeout."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass
        proc.wait()
        return 124, "", "TIMEOUT"


def _is_viable_seed(ghc_bin: str, path: Path) -> bool:
    """
    A candidate is viable if `ghc -fno-code` accepts it — type-checking
    only, same mechanism (and same flag) the driver itself uses, so
    "harvested" and "actually fuzzable" mean the same thing. Unlike a full
    build, -fno-code doesn't require `main` (only linking does).

    Checked in an isolated empty temp dir, NOT path.parent: _copy_kept
    flattens every kept candidate into a single shared seeds_dir, renaming
    it to avoid collisions, so a same-directory sibling module the file
    happens to import (e.g. nofib's real/gc/parallel benchmarks, which are
    mostly Main.hs + helper modules in one dir) will resolve here — under
    its original filename — but won't once copied (the sibling gets
    flattened under an unrelated prefixed name too, and `module Foo`
    requires a file literally named Foo.hs to be found via import search).
    Checking against path.parent silently overcounts exactly these
    multi-file cases as viable. This is also why real Hackage packages
    don't work as a harvest source: individual files there import
    *sibling* modules within the same package (not just external boot
    libraries), which a single-file check can't resolve — empirically
    measured at 3-10% yield on optparse-applicative/xmonad/aeson, an order
    of magnitude worse than GHC's testsuite (deliberately standalone) or
    Rosetta Code/Exercism (deliberately small, self-contained solutions).

    rc==0 is the sole bar (previously a stderr-substring marker list let
    some rc!=0 candidates through too, on the theory that only "well-known"
    failure text should disqualify a seed — dropped after finding a
    counter-example: nofib's spectral/hartel/*.hs use `{-# LANGUAGE CPP #-}`
    with a relative `#include "../Fast2haskell.hs"`, which fails with a gcc
    preprocessor "fatal error" no text marker was written to catch. A
    substring list can only ever cover failure shapes someone already
    thought to enumerate; rc==0 can't be circumvented that way).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / path.name
        shutil.copy2(path, tmp_path)
        rc, _out, _err = _run_with_group_kill(
            [ghc_bin, "-fno-code", "-v0", str(tmp_path)], cwd=tmp, timeout=_HARVEST_TIMEOUT,
        )
    return rc == 0  # rc==124 (timeout) and every other non-zero rc are both dropped


def _harvest_pool(ghc_bin: str, pool_list, cap: int, label: str):
    """Run the viability filter over pool_list concurrently, stopping once
    `cap` candidates have passed (cancelling remaining in-flight work)."""
    kept = []
    tested = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=_HARVEST_WORKERS) as pool:
        futures = {pool.submit(_is_viable_seed, ghc_bin, p): p for p in pool_list}
        for fut in concurrent.futures.as_completed(futures):
            tested += 1
            p = futures[fut]
            try:
                if fut.result():
                    kept.append(p)
            except Exception:
                pass
            if len(kept) >= cap:
                pool.shutdown(wait=False, cancel_futures=True)
                break
    print(f"  {label}: {len(kept)} kept from {tested} tried (pool={len(pool_list)}, cap={cap})")
    return kept


def _copy_kept(kept, root: Path, seeds_dir: Path, prefix: str) -> int:
    """Copy viable candidates into seeds_dir, flattening their path (relative
    to root) into the filename so identically-named files from different
    subdirectories/sources never collide."""
    copied = 0
    for p in kept:
        rel = p.relative_to(root)
        safe_name = f"{prefix}_" + str(rel).replace(os.sep, "__")
        dst = seeds_dir / safe_name
        if not dst.exists():
            try:
                shutil.copy2(p, dst)
                copied += 1
            except OSError:
                pass
    return copied


def _sparse_clone(url: str, dest: Path, patterns: list, cone: bool = False) -> bool:
    """
    Shallow, blob-filtered, sparse-checkout clone restricted to `patterns`.
    cone=True: patterns are plain directory paths, checked out recursively
    (simple, used for ghc/ghc's single testsuite/tests directory).
    cone=False: patterns are gitignore-style globs (supports 'Task/*/
    Haskell/*'-style matching), needed when the target paths are scattered
    across many sibling directories (Rosetta Code's per-task layout).
    """
    if dest.exists():
        return True
    print(f"Cloning {url} (shallow, sparse checkout of {patterns})...")
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--sparse", url, str(dest)],
            check=True,
        )
        if cone:
            subprocess.run(["git", "sparse-checkout", "set", *patterns], cwd=str(dest), check=True)
        else:
            subprocess.run(["git", "sparse-checkout", "init", "--no-cone"], cwd=str(dest), check=True)
            subprocess.run(["git", "sparse-checkout", "set", "--no-cone", *patterns], cwd=str(dest), check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Warning: could not clone {url} ({e}); skipping this source.")
        shutil.rmtree(dest, ignore_errors=True)
        return False


def _harvest_ghc_testsuite(project_root: Path, ghc_bin: str, seeds_dir: Path) -> int:
    """
    Clone ghc/ghc's testsuite (shallow, sparse-checkout of testsuite/tests
    only) and harvest a filtered sample of should_run/*.hs, should_compile/
    *.hs, and flat <category>/*.hs (e.g. th/, polykinds/, module/, gadt/,
    printer/ — ~90 categories, one level deep, not nested under a
    should_run/should_compile subdirectory) programs into seeds_dir —
    these are GHC's own regression tests, so most are small, self-contained,
    standard-library-only programs. Roughly half are named
    T<gitlab-issue-number>(suffix).hs — literal regression tests pinned to
    historic GHC bug reports. should_fail/*.hs is excluded (deliberately
    invalid, by design). A real, execution-verified filter (rather than
    static analysis) weeds out the multi-module tests that won't
    type-check standalone.

    should_compile was previously excluded because it was checked against
    `runghc`, which requires a runnable `main` (only ~4% have one — most
    should_compile fragments declare their own `module Foo where` and are
    library code, not programs). Once the driver switched to
    `ghc -fno-code` (type-check only, no `main` required), the same check
    applied to should_compile yields ~96%, so it's now included too.

    The flat <category>/*.hs glob covers th/ (Template Haskell — one of
    GHC's most bug-dense areas, compile-time metaprogramming) and every
    other category with the same layout (~2470 files total, ~59% pass the
    viability check, measured against 300 samples) in one pattern, since
    it's disjoint from */should_run|should_compile/*.hs (different path
    depth) — no double-counting. driver/, ghci/, ghc-api/, cabal/,
    haddock/ etc. are included in the attempt pool too but mostly fail
    (tooling/CLI-flag tests, not standalone programs) and get filtered out
    naturally by the same execution-verified check.
    """
    ghc_src = project_root / "ghc-src"
    if not _sparse_clone(
        "https://github.com/ghc/ghc.git", ghc_src, ["testsuite/tests"], cone=True,
    ):
        return 0

    tests_root = ghc_src / "testsuite" / "tests"
    candidates = (
        sorted(tests_root.glob("*/should_run/*.hs"))
        + sorted(tests_root.glob("*/should_compile/*.hs"))
        + sorted(tests_root.glob("*/*.hs"))
    )
    if not candidates:
        print("Warning: no should_run|should_compile|<category> *.hs files found under ghc-src testsuite.")
        return 0

    random.seed(0)  # deterministic harvest across re-runs
    ticketed = [p for p in candidates if _TICKET_NUMBERED_RE.match(p.name)]
    other = [p for p in candidates if not _TICKET_NUMBERED_RE.match(p.name)]
    random.shuffle(ticketed)
    random.shuffle(other)

    print(f"Testing ghc/ghc testsuite candidates with ghc -fno-code "
          f"(timeout={_HARVEST_TIMEOUT}s, workers={_HARVEST_WORKERS})...")
    kept = (
        _harvest_pool(ghc_bin, ticketed, _GHC_TICKET_CAP, "ticket-numbered (historic bugs)")
        + _harvest_pool(ghc_bin, other, _GHC_OTHER_CAP, "generically-named")
    )
    copied = _copy_kept(kept, tests_root, seeds_dir, "ghc")
    print(f"Harvested {copied} new ghc/ghc testsuite seeds ({len(kept)} passed viability).")
    return copied


def _harvest_rosetta_code(project_root: Path, ghc_bin: str, seeds_dir: Path) -> int:
    """
    Clone acmeism/RosettaCodeData (shallow, sparse-checkout of only the
    Haskell paths — the full repo interleaves every language per task and
    is 1.2GB; sparse-checkout keeps it to ~70MB) and harvest a filtered
    sample of small, idiomatic, real-world-style solutions to common
    programming tasks. These are stylistically different from GHC's own
    testsuite (which specifically targets compiler edge cases) — broader
    "does GHC handle normal user code" coverage rather than compiler-bug
    density. ~42% pass the viability check (measured against 300 of 2304
    Haskell solutions).
    """
    rosetta_src = project_root / "rosetta-src"
    if not _sparse_clone(
        "https://github.com/acmeism/RosettaCodeData.git", rosetta_src,
        ["Task/*/Haskell/**", "Lang/Haskell/**"],
    ):
        return 0

    candidates = sorted(rosetta_src.rglob("*.hs"))
    if not candidates:
        print("Warning: no .hs files found under rosetta-src.")
        return 0

    random.seed(1)
    random.shuffle(candidates)

    print(f"Testing Rosetta Code candidates with ghc -fno-code "
          f"(timeout={_HARVEST_TIMEOUT}s, workers={_HARVEST_WORKERS})...")
    kept = _harvest_pool(ghc_bin, candidates, _ROSETTA_CAP, "rosetta-code")
    copied = _copy_kept(kept, rosetta_src, seeds_dir, "rosetta")
    print(f"Harvested {copied} new Rosetta Code seeds ({len(kept)} passed viability).")
    return copied


def _harvest_exercism(project_root: Path, ghc_bin: str, seeds_dir: Path) -> int:
    """
    Clone exercism/haskell (small repo, ~11MB, full clone) and harvest the
    reference solutions under .meta/ — concept exercises keep theirs at
    .meta/exemplar/src/*.hs, practice exercises at
    .meta/examples/<variant>/src/*.hs. (exercises/*/src/*.hs, outside
    .meta, are learner-facing stub files with the implementation left
    blank — not valid standalone programs, so they're deliberately not
    globbed here.) Small pool (139 files) but high quality: ~86% pass the
    viability check.
    """
    exercism_src = project_root / "exercism-src"
    if not exercism_src.exists():
        print("Cloning exercism/haskell...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/exercism/haskell.git", str(exercism_src)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: could not clone exercism/haskell ({e}); skipping this source.")
            return 0

    candidates = sorted(exercism_src.glob("**/.meta/**/*.hs"))
    if not candidates:
        print("Warning: no .hs files found under exercism-src/**/.meta/.")
        return 0

    random.seed(2)
    random.shuffle(candidates)

    print(f"Testing Exercism candidates with ghc -fno-code "
          f"(timeout={_HARVEST_TIMEOUT}s, workers={_HARVEST_WORKERS})...")
    kept = _harvest_pool(ghc_bin, candidates, _EXERCISM_CAP, "exercism")
    copied = _copy_kept(kept, exercism_src, seeds_dir, "exercism")
    print(f"Harvested {copied} new Exercism seeds ({len(kept)} passed viability).")
    return copied


def _harvest_nofib(project_root: Path, ghc_bin: str, seeds_dir: Path) -> int:
    """
    Clone ghc/nofib (small, ~60MB, full clone — it's a separate repo from
    ghc/ghc, referenced there only as a submodule, so it isn't covered by
    the ghc/ghc sparse-checkout above) and harvest a filtered sample of its
    benchmark programs. Unlike the testsuite (compiler-edge-case density)
    or Rosetta/Exercism (small idiomatic solutions), nofib programs are
    performance benchmarks — different code shapes (tight loops, strict
    accumulators, array-heavy numeric code) that stress the optimizer
    differently. real/ in particular is almost entirely multi-file
    programs (Main.hs plus sibling modules in the same directory), so
    expect a lower yield here than the other sources; the execution-
    verified filter drops those the same way it drops multi-module
    Hackage files.
    """
    nofib_src = project_root / "nofib-src"
    if not nofib_src.exists():
        print("Cloning ghc/nofib...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "https://gitlab.haskell.org/ghc/nofib.git", str(nofib_src)],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: could not clone ghc/nofib ({e}); skipping this source.")
            return 0

    candidates = sorted(nofib_src.rglob("*.hs"))
    if not candidates:
        print("Warning: no .hs files found under nofib-src.")
        return 0

    random.seed(3)
    random.shuffle(candidates)

    print(f"Testing nofib candidates with ghc -fno-code "
          f"(timeout={_HARVEST_TIMEOUT}s, workers={_HARVEST_WORKERS})...")
    kept = _harvest_pool(ghc_bin, candidates, _NOFIB_CAP, "nofib")
    copied = _copy_kept(kept, nofib_src, seeds_dir, "nofib")
    print(f"Harvested {copied} new nofib seeds ({len(kept)} passed viability).")
    return copied


def _harvest_codenet(project_root: Path, ghc_bin: str, seeds_dir: Path) -> int:
    """
    Harvest IBM Project CodeNet's Haskell subset — ~14M competitive-judge
    submissions (AtCoder/AIZU) across 55 languages, of which the Haskell
    slice lives at Project_CodeNet/data/<problem>/Haskell/*.hs.

    Unlike the other sources, this isn't a git repo: it's a single 7.8GB
    tarball with no separate per-language or metadata-only download
    (IBM discontinued the standalone metadata tarball — the full archive
    is the only option: https://github.com/IBM/Project_CodeNet). This
    function does NOT download it — that's a one-time, multi-GB fetch
    unsuitable for running on every setup — it only harvests from
    `codenet-src/` if that directory has already been populated (e.g. via
    the maintainer streaming just the Haskell entries out of the tarball:
    `curl -sL <url> | tar -xzf - --wildcards 'Project_CodeNet/data/*/Haskell/*' --strip-components=2 -C codenet-src`).

    Submissions are competitive-programming solutions: single `Main.hs`-
    style files by construction (a judge compiles exactly one submitted
    file), so multi-module failures are expected to be rare here — the
    dominant failure mode instead should be missing common extensions
    (e.g. GHC2021-only syntax on an old default) or judge-specific stdin/
    stdout assumptions that don't affect standalone type-checking anyway.
    """
    codenet_src = project_root / "codenet-src"
    if not codenet_src.exists() or not any(codenet_src.iterdir()):
        print("Skipping CodeNet: codenet-src/ not present or empty "
              "(the 7.8GB Project CodeNet tarball must be fetched manually; see docstring).")
        return 0

    candidates = sorted(codenet_src.rglob("*.hs"))
    if not candidates:
        print("Warning: no .hs files found under codenet-src.")
        return 0

    random.seed(4)
    random.shuffle(candidates)

    print(f"Testing CodeNet candidates with ghc -fno-code "
          f"(timeout={_HARVEST_TIMEOUT}s, workers={_HARVEST_WORKERS})...")
    kept = _harvest_pool(ghc_bin, candidates, _CODENET_CAP, "codenet")
    copied = _copy_kept(kept, codenet_src, seeds_dir, "codenet")
    print(f"Harvested {copied} new CodeNet seeds ({len(kept)} passed viability).")
    return copied


def setup(project_root):
    """
    Sets up the Haskell fuzzing environment (runs inside the fuzz-haskell
    container):
    1. GHC ships pre-installed in the haskell:latest base image — just
       verify the toolchain.
    2. Seed the corpus from the small hand-curated set under
       curated_seeds/ (IORef/MVar/STM/State-monad examples exercising the
       constructs state fusion targets — note the driver only type-checks
       these, it never runs them, so state fusion's forkIO/MVar harness
       is exercised as compiler input, not as an actual concurrent race).
    3. Harvest a larger, execution-verified sample from five sources —
       ghc/ghc's own testsuite (compiler-bug-density-focused),
       Rosetta Code (broad real-world-style coverage), Exercism
       (small, high-quality reference solutions), nofib (ghc/nofib —
       performance benchmarks, a different code shape: tight loops,
       array-heavy numeric code), and IBM Project CodeNet's Haskell
       subset (competitive-judge submissions, single-file by
       construction). CodeNet's 7.8GB tarball isn't fetched by this
       script — see _harvest_codenet's docstring — it only harvests from
       codenet-src/ if already populated. Real Hackage packages were
       tried and rejected as a source: individual files there depend
       on sibling modules within the same package, which a single-file
       `ghc -fno-code` check can't resolve (measured 3-10% yield on
       optparse-applicative/xmonad/aeson vs. 42-96% for the other sources
       actually used) — supporting it properly would need multi-file/
       cabal-aware seeds, a much bigger change to the whole pipeline
       (parser/driver/fusion all assume one seed = one self-contained
       .hs file).
    """
    print(f"Setting up Haskell in: {project_root}")
    # Must be absolute: _is_viable_seed runs ghc with cwd=path.parent, so a
    # relative project_root would make the file argument resolve against
    # the wrong directory.
    project_root = Path(project_root).resolve()

    ghc = _resolve_ghc()
    try:
        result = subprocess.run(
            [ghc, "--version"], capture_output=True, text=True, check=True
        )
        print(f"ghc available: {result.stdout.strip()} ({ghc})")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: ghc not found: {e}")
        sys.exit(1)

    seeds_dir = project_root / "seeds"
    seeds_dir.mkdir(exist_ok=True)

    curated_dir = project_root / "curated_seeds"
    curated_copied = 0
    for src in sorted(curated_dir.glob("*.hs")):
        dst = seeds_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            curated_copied += 1
    print(f"Copied {curated_copied} new curated seeds "
          f"({len(list(curated_dir.glob('*.hs')))} total) into {seeds_dir}")

    for harvester in (_harvest_ghc_testsuite, _harvest_rosetta_code, _harvest_exercism,
                      _harvest_nofib, _harvest_codenet):
        try:
            harvester(project_root, ghc, seeds_dir)
        except Exception as e:
            print(f"Warning: {harvester.__name__} failed non-fatally: {e}")

    total = len(list(seeds_dir.glob("*.hs")))
    print(f"Seed corpus ready: {total} .hs files in {seeds_dir}")
    print("Haskell setup complete.")
