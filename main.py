import argparse
import logging
import os
import re
import sys
import subprocess
import time
import random
import importlib.util
import sqlite3
import json

# Add project root to path so we can import 'core' modules
sys.path.append(os.getcwd())

from core.orchestrator import FusionFuzzLoop
from core.config_loader import load_project_config
from core.fusion import get_strategies, Seed

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FFL.Main")

def filter_excluded_seeds(seeds, config):
    """Drop seeds matching any regex in config's paths.seed_exclude_patterns.

    A seed that cannot compile in *this* environment (a header the local
    toolchain never built, a target triple it lacks) or that is a negative
    test by construction (clang's `-verify` tests, whose code is
    deliberately ill-formed) can only ever produce invalid children, so
    every execution spent on it is wasted and it drags the fused-valid rate
    down without contributing reachable compiler paths.

    Entirely config-driven and off by default: a project with no
    seed_exclude_patterns key keeps every seed, exactly as before.
    """
    raw_patterns = (config.get("paths", {}) or {}).get("seed_exclude_patterns") or []
    if not raw_patterns:
        return seeds

    compiled = []
    for entry in raw_patterns:
        if isinstance(entry, dict):   # {pattern: ..., reason: ...}
            pattern, reason = entry.get("pattern"), entry.get("reason", "")
        else:
            pattern, reason = entry, ""
        if not pattern:
            continue
        try:
            compiled.append((re.compile(pattern), pattern, reason))
        except re.error as e:
            logger.warning(f"Ignoring invalid seed_exclude_pattern {pattern!r}: {e}")

    if not compiled:
        return seeds

    kept, dropped = [], {}
    for seed in seeds:
        content = seed.content or ""
        for rx, pattern, reason in compiled:
            if rx.search(content):
                key = reason or pattern
                dropped[key] = dropped.get(key, 0) + 1
                break
        else:
            kept.append(seed)

    if dropped:
        total = sum(dropped.values())
        logger.info(
            f"Excluded {total}/{len(seeds)} seeds via paths.seed_exclude_patterns "
            f"(remove them from {config.get('project_name')}'s config.yaml to keep these seeds):"
        )
        for key, count in sorted(dropped.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {count:6d}  {key}")
    return kept


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion Fuzz Loop (FFL)")
    parser.add_argument("--project", type=str, default=None, help="Project name (folder in projects/)")
    parser.add_argument("--iterations", type=int, default=-1, help="Fuzzing iterations")
    parser.add_argument("--time", type=float, default=None, metavar="SECONDS",
                        help="Wall-clock budget in seconds: stop after SECONDS instead of "
                             "running until --iterations / pairwise saturation / Ctrl-C. "
                             "Example: --time 3600 fuzzes for one hour. Combine with "
                             "--iterations and whichever limit is hit first stops the run. "
                             "The budget is checked when work is scheduled, so executions "
                             "already in flight are allowed to finish — a run can overshoot "
                             "by up to one execution timeout. Also applies to --save-to "
                             "(stops generating) and --execute (stops replaying).")
    parser.add_argument("--setup", action="store_true", default=False, help="Force project setup/seed parsing")
    parser.add_argument("--preprocessing", action="store_true", default=False,
                        help="[cpython only] Run projects/cpython/preprocessing.py's dynamic "
                             "tracing pass during --setup. Unrelated to --pre-analysis below "
                             "(this is a cpython-specific setup-time step; --pre-analysis is "
                             "the generic per-seed metadata pass used by every project's "
                             "fusion strategies).")
    parser.add_argument("--bug-corpus", action="store_true", default=False,
                        help="Seed from ./corpus/corpus.db: inject pre-translated bug reproducers into the project corpus")
    parser.add_argument("--sample-log", type=str, default=None, nargs="?",
                        const="output/{project}_samples.log",  # value when flag given without arg
                        metavar="PATH",
                        help="Log every sample's seed content + stdout/stderr. "
                             "Omit PATH to use default: output/<project>_samples.log")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Execute every seed once before fuzzing and filter the corpus down "
                             "to seeds whose own execution returns rc==0. Does not collect fusion "
                             "metadata by itself — pair with --pre-analysis for that; combined, "
                             "they share one execution pass per seed rather than running it twice.")
    parser.add_argument("--pre-analysis", action="store_true", default=False,
                        help="Execute every seed once before fuzzing (probe-instrumented where "
                             "a language needs one) and collect the metadata fusion strategies "
                             "use at runtime: dataflow graphs, declared/observed variable types, "
                             "and core/state_analysis.py's live-variable most-complex-state points "
                             "for state fusion. "
                             "Does not filter the corpus by validity — pair with --dry-run for "
                             "that. Required for --declaration-fusion/--struct-fusion (disabled "
                             "otherwise, with a warning). Without it, --dataflow-fusion falls back "
                             "to a lightweight on-the-fly random-variable-connect rule and "
                             "--state-fusion falls back to picking any random line, for every "
                             "project.")
    parser.add_argument("--guided-fusion", action="store_true", default=False,
                        help="REMOVED — accepted and ignored. Producer-consumer guided parent "
                             "selection (prefer a pair where one seed produces a resource the "
                             "other consumes) has been dropped: it cost throughput without a "
                             "measured payoff. Parent selection is now always a uniformly random "
                             "pair among those not yet fused.")
    parser.add_argument("--concurrency", type=int, default=None, help="Override the number of threads for execution (default is from config.yaml)")
    parser.add_argument("--reduce", type=str, default=None, metavar="BUG_DIR",
                        help="Minimize a crash reproducer (test.<ext>) to min.<ext> using delta "
                             "debugging, then update the bug report. "
                             "Example: --reduce ./output/bugs/php/Assertion__0_2f95bf0e")
    parser.add_argument("--signature", type=str, default=None, metavar="SIG",
                        help="Override the crash signature used by --reduce. "
                             "Any crash whose output contains this string counts. "
                             'Example: --signature "core dumped"')
    parser.add_argument("--setup-cov", action="store_true", default=False,
                        help="Build project with gcov (no sanitizers) for coverage measurement")
    parser.add_argument("--gcov", action="store_true", default=False,
                        help="After fuzzing, collect and print gcov line coverage information")
    parser.add_argument("--dataflow-fusion", action="store_true", default=False,
                        help="Enable dataflow fusion: bridge variable linking via def-use graph. "
                             "Each fusion randomly picks A->B or B->A direction. Combinable with "
                             "--state-fusion/--declaration-fusion — every technique you enable is "
                             "added to the same pool, and each pool member independently has a "
                             "--fusion-rate chance (default 80%%) of being applied per iteration "
                             "(never zero — one is picked at random as a fallback if every draw "
                             "comes up empty). "
                             "If none of --dataflow-fusion/--state-fusion/--declaration-fusion are "
                             "given, dataflow fusion is enabled by default. Without --pre-analysis, "
                             "falls back to a lightweight on-the-fly rule for every project: scan "
                             "each side's variables straight from its source text and connect one "
                             "uniformly random pair (no dependency graph, no type awareness).")
    parser.add_argument("--struct-fusion", action="store_true", default=False,
                        help="[rust only] Item-level fusion: nest struct/enum/trait/impl/fn "
                             "definitions from one seed inside a container (mod/fn body) found "
                             "in the other, plus supertrait injection, impl grafting, and "
                             "generic bound injection. Does not use statement/dataflow fusion. "
                             "Alias of --declaration-fusion for rust. Requires --pre-analysis "
                             "(disabled otherwise, with a warning).")
    parser.add_argument("--state-fusion", action="store_true", default=False,
                        help="[php/cpython/clang/flang/lfortran/swift/haskell/mlir/naga] Enable the state-of-"
                             "interest-driven fusion strategy (core/state_analysis.py): profiles "
                             "each seed for the point with the most live, still-in-scope variables, "
                             "then grafts one seed's continuation into the other's state at that "
                             "point instead of only bridging a single value. Combinable with "
                             "--dataflow-fusion/--declaration-fusion — every technique you enable is "
                             "added to the same pool, and each pool member independently has a "
                             "--fusion-rate chance (default 80%%) of being applied per iteration "
                             "(never zero — one is picked at random as a fallback if every draw "
                             "comes up empty). "
                             "Without --pre-analysis, falls back to picking any random line as the "
                             "splice point instead. "
                             "[haskell] always picks a random line regardless of --pre-analysis — "
                             "its layout-based scoping doesn't fit the live-variable-count rule. "
                             "[haskell/mlir] least/moderately verified — no local ghc/mlir-opt "
                             "toolchain to compile-check against in this repo's dev environment; "
                             "clang/cpython were checked against g++ and ast.parse respectively.")
    parser.add_argument("--declaration-fusion", action="store_true", default=False,
                        help="Enable declaration fusion: confuse declarations rather than "
                             "dataflow by making an extensible declaration expression in one seed "
                             "(base class list, template parameter, generic/trait/superclass "
                             "bound, operand type constraint) refer to a declaration in the other "
                             "seed. Triggers on declare/compile alone, no runtime dataflow needed. "
                             "Combinable with --dataflow-fusion/--state-fusion — every technique "
                             "you enable is added to the same pool, and each pool member "
                             "independently has a --fusion-rate chance (default 80%%) of being "
                             "applied per iteration (never zero — one is picked at random as a "
                             "fallback if every draw comes up empty). Requires --pre-analysis "
                             "(disabled otherwise, with a "
                             "warning — not considered feasible without its richer per-seed "
                             "metadata). "
                             "[rust] alias of --struct-fusion (item nesting, supertrait/impl/"
                             "bound injection). [clang] base class + template param injection, "
                             "item nesting. [php] implements/extends + trait-use injection. "
                             "[swift] protocol conformance injection. [cpython] extra base-class "
                             "injection (MRO/metaclass conflicts at class-statement time). "
                             "[haskell] typeclass superclass constraint injection. [flang/"
                             "lfortran] derived-type EXTENDS() injection. [mlir] function-signature "
                             "operand/result type swap. [naga] WGSL struct/member/function type "
                             "reference injection (verified structurally only, no compiler in "
                             "this repo's dev environment for haskell/flang/swift/mlir/php/naga).")
    parser.add_argument("--children-per-pair", type=int, default=2, metavar="N",
                        help="How many fused programs to produce from each selected parent "
                             "pair per iteration (default 2 = one bidirectional draw, the "
                             "original behaviour). Higher values amortise parent selection "
                             "and the strategy chain's prefix over more children, so the "
                             "per-child fusion cost drops (measured on clang: 2.39 ms at "
                             "N=2, 1.70 ms at N=8). The children of one pair share that "
                             "prefix and are therefore correlated — raising N buys raw "
                             "throughput at the cost of sample diversity and slower pairwise "
                             "coverage, so measure new-bug rate, not just tests/s.")
    parser.add_argument("--fusion-rate", type=float, default=0.8, metavar="P",
                        help="Probability (0.0-1.0) that each enabled fusion technique in the "
                             "pool is independently applied to a given parent pair per iteration "
                             "(core/orchestrator.py's FusionFuzzLoop._pick_strategy_chain). If "
                             "every draw comes up empty, one technique is picked uniformly at "
                             "random as a fallback, so an iteration is never left untouched. "
                             "Has no effect with a pool of size 1 (e.g. --dataflow-fusion passed "
                             "alone), which always applies that one technique. Default: 0.8.")
    parser.add_argument("--corpus-size", type=int, default=None, metavar="N",
                        help="Sample N seeds from the loaded corpus for fusion "
                             "instead of using all seed programs")
    parser.add_argument("--diverse", action="store_true", default=False,
                        help="With --corpus-size, select dissimilar seeds (best-effort "
                             "greedy farthest-point sampling) instead of a uniform random sample")
    parser.add_argument("--save-subset", type=str, default=None, metavar="PATH",
                        help="Save the selected corpus subset (after --corpus-size/--diverse) "
                             "to PATH for reuse via --load-subset")
    parser.add_argument("--save-to", type=str, default=None, metavar="OUTPUT_DIR",
                        help="Fusion-only mode: don't fuzz. Run program fusion exactly as "
                             "fuzzing would (same corpus, same parent selection, same "
                             "--dataflow-fusion/--state-fusion/--declaration-fusion pool and "
                             "--fusion-rate), but write each fused program to OUTPUT_DIR as "
                             "fused_<n>.<ext> instead of executing it. Every ordered pair of "
                             "corpus seeds is fused exactly once — (a,b) and (b,a) are separate "
                             "programs, and (a,a) too unless --no-self-fusion — so N seeds give "
                             "exactly N*N files (e.g. --corpus-size 100 -> 10000). --iterations N "
                             "caps how many are written (default -1: all pairs), taken from a "
                             "shuffled pair order so a capped run still spans the whole corpus. "
                             "OUTPUT_DIR/manifest.jsonl records each program's ordered host/donor "
                             "parents and the strategies applied. Compatible with --pre-analysis/"
                             "--dry-run/--corpus-size, which still run beforehand; --gcov/"
                             "--sample-log are ignored.")
    parser.add_argument("--execute", type=str, default=None, metavar="PROGRAM_DIR",
                        help="Replay mode: don't fuse or generate anything — just run every "
                             "program in PROGRAM_DIR (recursively; manifest.jsonl and hidden "
                             "files skipped) through the project driver and report bugs, with "
                             "the same crash triage, deduplication and output/bugs/<project>/ "
                             "bundles as fuzzing. Pairs with --save-to: generate once, execute "
                             "the folder later. Only --project is required; --concurrency and "
                             "--sample-log also apply, every other flag is ignored. "
                             "Example: python3 main.py --project php --execute ./test100")
    parser.add_argument("--no-self-fusion", action="store_true", default=False,
                        help="With --save-to, skip the (a,a) self-fusion pairs, giving "
                             "N*(N-1) programs instead of N*N.")
    parser.add_argument("--c", dest="lang_c", action="store_true", default=False,
                        help="[clang only] Enable C. Combinable with --cpp/--m; pass none of "
                             "the three to get every language (the default, unchanged). "
                             "The selection restricts BOTH which seed files are parsed into "
                             "the corpus (--setup) / loaded from it, and which languages fused "
                             "programs may be written in. Example: --c --cpp parses .c/.cpp/.cc/"
                             ".cxx seeds and emits .c and .cpp programs.")
    parser.add_argument("--cpp", dest="lang_cpp", action="store_true", default=False,
                        help="[clang only] Enable C++ (.cpp/.cc/.cxx). See --c.")
    parser.add_argument("--m", dest="lang_objc", action="store_true", default=False,
                        help="[clang only] Enable Objective-C (.m). See --c. Objective-C++ "
                             "(.mm) is only used when --cpp and --m are both given, since a "
                             ".mm program is compilable as neither language alone.")
    parser.add_argument("--load-subset", type=str, default=None, metavar="PATH",
                        help="Load a previously saved corpus subset from PATH instead of "
                             "the project corpus (skips --corpus-size/--diverse selection)")

    args = parser.parse_args()

    if args.time is not None and args.time <= 0:
        parser.error(f"--time must be positive (got {args.time})")

    if args.guided_fusion:
        logger.warning(
            "--guided-fusion has been removed and is ignored: parent selection is "
            "always a uniformly random pair among those not yet fused."
        )

    # --reduce: standalone mode, project inferred from path
    if args.reduce:
        from core.reducer import reduce_command
        reduce_command(args.reduce, override_sig=args.signature)
        sys.exit(0)
    
    if not args.project:
        parser.error("--project is required for fuzzing mode")

    # 1. Load Configuration
    try:
        config = load_project_config(args.project)
        logger.info(f"Loaded configuration for {args.project}")
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)
        
    # --- Language selection (--c/--cpp/--m, clang only) ---------------
    # Empty set == no restriction (every language), so the default
    # behavior is exactly what it was before these flags existed.
    clang_langs = set()
    if args.lang_c:
        clang_langs.add("c")
    if args.lang_cpp:
        clang_langs.add("cpp")
    if args.lang_objc:
        clang_langs.add("objc")
    if clang_langs and args.project != "clang":
        logger.warning(
            f"--c/--cpp/--m only apply to --project clang; ignoring them for "
            f"--project {args.project}."
        )
        clang_langs = set()

    def _load_parser_module(project):
        parser_path = os.path.join("projects", project, "parser.py")
        spec = importlib.util.spec_from_file_location("project_parser", parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if clang_langs and hasattr(module, "set_languages"):
            module.set_languages(clang_langs)
        return module

    if clang_langs:
        logger.info(f"Languages restricted to: {', '.join(sorted(clang_langs))}")

    # Apply command-line config overrides
    if args.concurrency is not None:
        if "execution" not in config or not isinstance(config.get("execution"), dict):
            config["execution"] = {}
        config["execution"]["concurrency"] = args.concurrency
        logger.info(f"Overriding execution concurrency to {args.concurrency}")

    # === REPLAY MODE (--execute): run pre-generated programs, no fusion ===
    # Handled before any corpus/setup work — replay needs neither a corpus
    # nor fusion strategies, just the project's driver.
    if args.execute:
        sample_log = args.sample_log.replace("{project}", args.project) if args.sample_log else None
        replayer = FusionFuzzLoop(config=config, strategies=[], initial_corpus=[])
        replayer.execute_folder(args.execute, sample_log=sample_log, max_seconds=args.time)
        sys.exit(0)

    # === BUG CORPUS MODE ===
    # Maps project name to canonical language key stored in corpus translations JSON
    _LANG_MAP = {
        "cpython": "python", "gcc": "c", "clang": "c",
        "go": "go", "rust": "rust", "php": "php",
        "swift": "swift", "lean": "lean", "mlir": "mlir",
        "naga": "rust", "wgslc": "wgsl", "sql": "sql",
        "lfortran": "flang",
    }
    _tgt_lang = _LANG_MAP.get(args.project.lower(), args.project.lower())

    if args.bug_corpus:
        bug_corpus_db = os.path.join("corpus", "corpus.db")
        if not os.path.exists(bug_corpus_db):
            logger.error(f"Bug corpus DB not found at {bug_corpus_db}. Run corpus/main.py first.")
            sys.exit(1)

        project_corpus_path = os.path.join("projects", args.project, "corpus.db")
        logger.info(f"Injecting bug corpus translations ({_tgt_lang}) → {project_corpus_path}")

        try:
            src_conn = sqlite3.connect(bug_corpus_db)
            rows = src_conn.execute(
                "SELECT id, project, name, translations FROM corpus WHERE translations != '{}'",
            ).fetchall()
            src_conn.close()
        except Exception as e:
            logger.error(f"Failed to read bug corpus: {e}")
            sys.exit(1)

        # Ensure target seeds DB exists (create if missing)
        dst_conn = sqlite3.connect(project_corpus_path)
        dst_conn.execute("""
            CREATE TABLE IF NOT EXISTS seeds (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                identifier TEXT UNIQUE,
                content    TEXT,
                metadata   TEXT
            )
        """)
        dst_conn.commit()
        # Add identifier column if the table pre-existed without it
        existing_cols = {row[1] for row in dst_conn.execute("PRAGMA table_info(seeds)")}
        if "identifier" not in existing_cols:
            dst_conn.execute("ALTER TABLE seeds ADD COLUMN identifier TEXT")
            dst_conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_seeds_identifier ON seeds(identifier)")
            dst_conn.commit()

        added = skipped = 0
        for row in rows:
            trans = json.loads(row[3])
            code = trans.get(_tgt_lang)
            if not code:
                continue
            identifier = f"bug_corpus_{row[0]}_{row[1]}_{row[2] or row[0]}"
            try:
                dst_conn.execute(
                    "INSERT INTO seeds (identifier, content, metadata) VALUES (?, ?, ?)",
                    (identifier, code, json.dumps({
                        "type": "bug_corpus",
                        "source_project": row[1],
                        "source_name": row[2],
                        "bug_corpus_id": row[0],
                    })),
                )
                added += 1
            except sqlite3.IntegrityError:
                skipped += 1
        dst_conn.commit()
        dst_conn.close()
        logger.info(f"Bug corpus: {added} seeds injected, {skipped} already present")

    # === STANDARD FUZZING SETUP ===
    
    # Determine Corpus Path
    project_corpus_path = os.path.join("projects", args.project, "corpus.db")
    
    # Auto-detect if setup is needed
    should_run_setup = args.setup or args.setup_cov or not os.path.exists(project_corpus_path)

    if should_run_setup:
        logger.info("Initializing Corpus (Setup Mode)...")
        
        # Save original CWD to restore after external scripts
        original_cwd = os.getcwd()
        
        # Step 1: Call setup.py first
        setup_script_path = os.path.join("projects", args.project, "setup.py")
        
        if os.path.exists(setup_script_path):
            logger.info(f"Found setup.py, loading module: {setup_script_path}")
            try:
                spec = importlib.util.spec_from_file_location("project_setup", setup_script_path)
                setup_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(setup_module)
                
                project_root = os.path.abspath(os.path.join("projects", args.project))
                if args.setup_cov and hasattr(setup_module, "setup_cov"):
                    logger.info(f"Executing setup_cov() with root: {project_root}")
                    try:
                        setup_module.setup_cov(project_root)
                        logger.info("Setup (gcov) function finished successfully.")
                    finally:
                        os.chdir(original_cwd)
                elif hasattr(setup_module, "setup"):
                    logger.info(f"Executing setup() with root: {project_root}")
                    try:
                        setup_module.setup(project_root)
                        logger.info("Setup function finished successfully.")
                    finally:
                        os.chdir(original_cwd)
                else:
                    logger.warning("setup.py found but no 'setup(project_root)' function defined.")
            except Exception as e:
                logger.error(f"Error executing setup function in setup.py: {e}")
                sys.exit(1)
        else:
            logger.info(f"No setup.py found at {setup_script_path}, skipping execution step.")

        # Step 1.5: Reflection
        reflection_script_path = os.path.join("projects", args.project, "reflection.py")
        if os.path.exists(reflection_script_path):
            logger.info(f"Found reflection.py, loading module: {reflection_script_path}")
            try:
                spec = importlib.util.spec_from_file_location("project_reflection", reflection_script_path)
                reflect_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(reflect_module)
                
                if hasattr(reflect_module, "reflect"):
                    project_root = os.path.abspath(os.path.join("projects", args.project))
                    logger.info(f"Executing reflect() with root: {project_root}")
                    try:
                        reflect_module.reflect(project_root)
                        logger.info("Reflection function finished successfully.")
                    finally:
                        os.chdir(original_cwd)
            except Exception as e:
                logger.error(f"Error executing reflection function: {e}")

        # Step 2: Seed Collection
        parser_path = os.path.join("projects", args.project, "parser.py")
        if not os.path.exists(parser_path):
            logger.error(f"Parser not found at {parser_path}")
            sys.exit(1)
            
        module = _load_parser_module(args.project)

        logger.info(f"Collecting seeds from {config['paths']['seed_source']}...")
        seed_blacklist = config.get("paths", {}).get("seed_blacklist", [])
        corpus_path = module.collect_seeds(config["paths"]["seed_source"], blacklist=seed_blacklist)
        
        if not corpus_path or not os.path.exists(corpus_path):
            logger.warning("No seeds found or corpus creation failed!")
            sys.exit(1)
        else:
            logger.info(f"Project corpus ready at: {corpus_path}")

        # Step 3: Preprocessing
        if args.preprocessing:
            logger.info("Starting Preprocessing...")
            preprocess_path = os.path.join("projects", args.project, "preprocessing.py")
            if os.path.exists(preprocess_path):
                # ... (preprocessing logic)
                try:
                    spec = importlib.util.spec_from_file_location("project_preprocess", preprocess_path)
                    prep_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(prep_module)
                    if hasattr(prep_module, "preprocess"):
                        project_root = os.path.abspath(os.path.join("projects", args.project))
                        try:
                            prep_module.preprocess(project_root)
                            logger.info("Preprocessing finished.")
                        finally:
                            os.chdir(original_cwd)
                except Exception as e:
                    logger.error(f"Preprocessing error: {e}")
            else:
                logger.warning("Preprocessing requested but no script found.")

    # 4. Load Corpus into Memory
    if not os.path.exists(project_corpus_path):
        logger.error(f"Corpus DB not found at {project_corpus_path}. Setup failed.")
        sys.exit(1)

    if args.load_subset:
        from core.corpus_sampling import load_subset
        logger.info(f"Loading saved corpus subset from {args.load_subset}...")
        initial_corpus = load_subset(args.load_subset)
        logger.info(f"Loaded {len(initial_corpus)} seeds from subset (skipping --corpus-size/--diverse).")
        initial_corpus = filter_excluded_seeds(initial_corpus, config)
        if clang_langs:
            # A saved subset may have been sampled with a different (or no)
            # language selection, so apply the filter here too.
            _mod = _load_parser_module(args.project)
            _allowed = set(_mod.allowed_extensions(clang_langs))
            _before = len(initial_corpus)
            initial_corpus = [
                s for s in initial_corpus
                if (s.metadata.get("extension") or ".c").lower() in _allowed
            ]
            if len(initial_corpus) != _before:
                logger.info(
                    f"Kept {len(initial_corpus)}/{_before} subset seeds matching "
                    f"{', '.join(sorted(clang_langs))}."
                )
    else:
        logger.info(f"Loading corpus from {project_corpus_path}...")

        module = _load_parser_module(args.project)

        raw_seeds = module.load_corpus(project_corpus_path)

        if not args.bug_corpus:
            _before = len(raw_seeds)
            raw_seeds = [s for s in raw_seeds if s["metadata"].get("type") != "bug_corpus"]
            _skipped = _before - len(raw_seeds)
            if _skipped:
                logger.info(
                    f"Excluding {_skipped} previously-injected bug corpus seeds "
                    f"from {project_corpus_path} (pass --bug-corpus to include them)."
                )

        initial_corpus = [
            Seed(content=s["content"], metadata={**s["metadata"], "filename": s["filename"]})
            for s in raw_seeds
        ]

        logger.info(f"Loaded {len(initial_corpus)} seeds into memory.")

        initial_corpus = filter_excluded_seeds(initial_corpus, config)

        # 4.5. Optionally sample down to a fixed corpus size
        if args.corpus_size is not None:
            if args.corpus_size < len(initial_corpus):
                if args.diverse:
                    from core.corpus_sampling import select_diverse_seeds
                    initial_corpus = select_diverse_seeds(initial_corpus, args.corpus_size)
                    logger.info(
                        f"Diversity-sampled {len(initial_corpus)} seeds "
                        f"(--corpus-size {args.corpus_size} --diverse) for fusion."
                    )
                else:
                    initial_corpus = random.sample(initial_corpus, args.corpus_size)
                    logger.info(f"Sampled {len(initial_corpus)} seeds (--corpus-size {args.corpus_size}) for fusion.")
            else:
                logger.info(
                    f"--corpus-size {args.corpus_size} >= loaded corpus size {len(initial_corpus)}; "
                    "using all seeds."
                )

        if args.save_subset:
            from core.corpus_sampling import save_subset
            save_subset(initial_corpus, args.save_subset)
            logger.info(f"Saved {len(initial_corpus)} selected seeds to {args.save_subset} for reuse via --load-subset.")

    # 5. Corpus pre-run pass: execute seeds that need it once, optionally
    #    filtering to rc=0 (--dry-run) and/or collecting rich fusion
    #    metadata (--pre-analysis), persisting either to the project corpus
    #    DB. Only runs when at least one of the two flags is passed;
    #    otherwise all loaded seeds are used as-is.
    if not initial_corpus:
        logger.error(
            "Corpus is empty after loading"
            + (f" (no seeds match --c/--cpp/--m selection: "
               f"{', '.join(sorted(clang_langs))})." if clang_langs else ".")
        )
        sys.exit(1)

    _max_workers = config.get("execution", {}).get("concurrency", 4)

    if args.dry_run or args.pre_analysis:
        from core.driver import get_driver
        from core.dryrun import run_dryrun_with_metadata

        logger.info(
            f"Corpus pre-run pass: {len(initial_corpus)} seeds "
            f"(dry-run={args.dry_run}, pre-analysis={args.pre_analysis}, "
            f"timeout=5s, workers={_max_workers})"
        )
        _valid_corpus = run_dryrun_with_metadata(
            seeds            = initial_corpus,
            driver_factory   = lambda: get_driver(config),
            db_path          = project_corpus_path,
            concurrency      = _max_workers,
            timeout          = 5,
            force            = args.setup,
            collect_metadata = args.pre_analysis,
            filter_valid     = args.dry_run,
            project_name     = args.project,
        )
        logger.info(
            f"Using {len(_valid_corpus)}/{len(initial_corpus)} seeds for fuzzing"
            + (" (filtered to rc=0)." if args.dry_run else " (unfiltered).")
        )
    else:
        _valid_corpus = initial_corpus
        logger.info(
            f"Using all {len(_valid_corpus)} seeds for fuzzing "
            "(pass --dry-run to filter, --pre-analysis to collect fusion metadata)."
        )

    # 6. Initialize & Run Orchestrator
    _strategies = get_strategies(args.project,
                                 dataflow_fusion=args.dataflow_fusion,
                                 struct_fusion=args.struct_fusion,
                                 declaration_fusion=args.declaration_fusion,
                                 state_fusion=args.state_fusion,
                                 pre_analysis_enabled=args.pre_analysis,
                                 clang_langs=clang_langs or None)
    if not _strategies:
        requested = [name for flag, name in [
            (args.dataflow_fusion, "--dataflow-fusion"),
            (args.declaration_fusion, "--declaration-fusion"),
            (args.state_fusion, "--state-fusion"),
            (args.struct_fusion, "--struct-fusion"),
        ] if flag]
        logger.error(
            f"No fusion strategy available for --project {args.project} with "
            f"{' '.join(requested) or '(default)'} — this project doesn't support "
            "the requested technique(s). See --help for which flags each project supports."
        )
        sys.exit(1)

    fuzzer = FusionFuzzLoop(
        config=config,
        strategies=_strategies,
        initial_corpus=_valid_corpus,
        pre_analysis_enabled=args.pre_analysis,
        fusion_rate=args.fusion_rate,
        children_per_pair=args.children_per_pair,
    )
    
    # === FUSION-ONLY MODE (--save-to): generate and save, never execute ===
    if args.save_to:
        fuzzer.generate_only(args.save_to, max_programs=args.iterations,
                             self_fusion=not args.no_self_fusion,
                             max_seconds=args.time)
        sys.exit(0)

    # === GCOV RESET (before fuzzing) ===
    if args.gcov:
        php_src_dir = os.path.join("projects", args.project, "php-src")
        if os.path.isdir(php_src_dir):
            logger.info("Resetting gcov counters (deleting .gcda files)...")
            subprocess.run(
                ["find", php_src_dir, "-name", "*.gcda", "-delete"],
                check=False
            )

    sample_log = args.sample_log.replace("{project}", args.project) if args.sample_log else None
    fuzzer.run(max_iterations=args.iterations, sample_log=sample_log, max_seconds=args.time)

    # === GCOV COVERAGE COLLECTION ===
    if args.gcov:
        php_src_dir = os.path.join("projects", args.project, "php-src")
        if not os.path.isdir(php_src_dir):
            logger.error(f"Cannot collect gcov data: {php_src_dir} not found")
            sys.exit(1)

        logger.info("Collecting gcov line coverage...")
        try:
            gcov_result = subprocess.run(
                ["sh", "-c", f"cd {php_src_dir} && find . -name '*.gcda' | head -1"],
                capture_output=True, text=True
            )
            if not gcov_result.stdout.strip():
                logger.warning("No .gcda files found. Was PHP built with --enable-gcov (--setup-cov)?")
            else:
                result = subprocess.run(
                    ["sh", "-c", f"""
cd {php_src_dir}
find . -name '*.gcno' -printf '%h\\n' | sort -u | while read dir; do
    (cd "$dir" && gcov -n *.gcno 2>/dev/null)
done
"""],
                    capture_output=True, text=True, timeout=300
                )
                total_lines = 0
                exec_lines = 0
                for line in result.stdout.splitlines():
                    m = re.match(r"Lines executed:(\d+\.\d+)% of (\d+)", line)
                    if m:
                        pct = float(m.group(1))
                        n = int(m.group(2))
                        exec_lines += int(pct * n / 100)
                        total_lines += n

                if total_lines > 0:
                    overall_pct = exec_lines / total_lines * 100
                    print(f"\n{'='*60}")
                    print(f"GCOV Line Coverage Summary")
                    print(f"{'='*60}")
                    print(f"  Lines executed: {exec_lines:,} / {total_lines:,} ({overall_pct:.2f}%)")
                    print(f"{'='*60}\n")
                else:
                    logger.warning("No gcov coverage data could be parsed.")

                if result.stderr:
                    for err_line in result.stderr.strip().splitlines()[:5]:
                        logger.debug(f"gcov stderr: {err_line}")

        except subprocess.TimeoutExpired:
            logger.error("gcov collection timed out after 300s")
        except FileNotFoundError:
            logger.error("gcov not found. Install gcc/gcov.")
