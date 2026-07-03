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

def get_db_content(db_path, identifier):
    """Helper to fetch content by identifier from a seed DB."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        res = cursor.execute("SELECT content FROM seeds WHERE identifier = ?", (identifier,)).fetchone()
        conn.close()
        return res[0] if res else None
    except Exception:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fusion Fuzz Loop (FFL)")
    parser.add_argument("--project", type=str, default=None, help="Project name (folder in projects/)")
    parser.add_argument("--iterations", type=int, default=-1, help="Fuzzing iterations")
    parser.add_argument("--setup", action="store_true", default=False, help="Force project setup/seed parsing")
    parser.add_argument("--preprocessing", action="store_true", default=False, help="Run seed preprocessing (dynamic info collection)")
    parser.add_argument("--bug-corpus", action="store_true", default=False,
                        help="Seed from ./corpus/corpus.db: inject pre-translated bug reproducers into the project corpus")
    parser.add_argument("--sample-log", type=str, default=None, nargs="?",
                        const="output/{project}_samples.log",  # value when flag given without arg
                        metavar="PATH",
                        help="Log every sample's seed content + stdout/stderr. "
                             "Omit PATH to use default: output/<project>_samples.log")
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Execute every seed once before fuzzing to collect metadata and filter out non-zero-RC seeds")
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
    parser.add_argument("--statement-fusion", action="store_true", default=False,
                        help="Enable statement fusion (dependency-graph interleave). "
                             "Each fusion randomly picks A->B or B->A direction.")
    parser.add_argument("--dataflow-fusion", action="store_true", default=False,
                        help="Enable dataflow fusion (bridge variable linking). "
                             "Each fusion randomly picks A->B or B->A direction.")
    parser.add_argument("--all-fusion", action="store_true", default=False,
                        help="Generate all 4 fusion variants per pair (stmt A→B, stmt B→A, "
                             "dataflow A+B, dataflow B+A). Each pair counts as one iteration.")
    parser.add_argument("--corpus-size", type=int, default=None, metavar="N",
                        help="Sample N seeds from the loaded corpus for fusion "
                             "instead of using all seed programs")
    parser.add_argument("--diverse", action="store_true", default=False,
                        help="With --corpus-size, select dissimilar seeds (best-effort "
                             "greedy farthest-point sampling) instead of a uniform random sample")
    parser.add_argument("--save-subset", type=str, default=None, metavar="PATH",
                        help="Save the selected corpus subset (after --corpus-size/--diverse) "
                             "to PATH for reuse via --load-subset")
    parser.add_argument("--load-subset", type=str, default=None, metavar="PATH",
                        help="Load a previously saved corpus subset from PATH instead of "
                             "the project corpus (skips --corpus-size/--diverse selection)")

    args = parser.parse_args()

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
        
    # Apply command-line config overrides
    if args.concurrency is not None:
        if "execution" not in config or not isinstance(config.get("execution"), dict):
            config["execution"] = {}
        config["execution"]["concurrency"] = args.concurrency
        logger.info(f"Overriding execution concurrency to {args.concurrency}")

    # 2. Connect to Database
    # from core.database import SeedStorage
    # db_path = f"output/{args.project}.db"
    # storage = SeedStorage(db_path)

    # === BUG CORPUS MODE ===
    # Maps project name to canonical language key stored in corpus translations JSON
    _LANG_MAP = {
        "cpython": "python", "gcc": "c", "clang": "c",
        "go": "go", "rust": "rust", "php": "php",
        "swift": "swift", "lean": "lean", "mlir": "mlir",
        "naga": "rust", "wgslc": "wgsl", "sql": "sql",
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
            
        spec = importlib.util.spec_from_file_location("project_parser", parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
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
    else:
        logger.info(f"Loading corpus from {project_corpus_path}...")

        parser_path = os.path.join("projects", args.project, "parser.py")
        spec = importlib.util.spec_from_file_location("project_parser", parser_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        raw_seeds = module.load_corpus(project_corpus_path)

        initial_corpus = [
            Seed(content=s["content"], metadata={**s["metadata"], "filename": s["filename"]})
            for s in raw_seeds
        ]

        logger.info(f"Loaded {len(initial_corpus)} seeds into memory.")

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

    # 5. Corpus dry-run: execute every seed once, collect rich metadata,
    #    persist it to the project corpus DB, and keep only rc=0 seeds.
    #    Only runs when --dry-run is passed; otherwise all loaded seeds are used.
    _max_workers = config.get("execution", {}).get("concurrency", 4)

    if args.dry_run:
        from core.driver import get_driver
        from core.dryrun import run_dryrun_with_metadata

        logger.info(
            f"Corpus dry-run: {len(initial_corpus)} seeds "
            f"(timeout=5s, workers={_max_workers})"
        )
        _valid_corpus = run_dryrun_with_metadata(
            seeds          = initial_corpus,
            driver_factory = lambda: get_driver(config),
            db_path        = project_corpus_path,
            concurrency    = _max_workers,
            timeout        = 5,
            force          = args.setup,
        )
        logger.info(f"Using {len(_valid_corpus)}/{len(initial_corpus)} valid seeds for fuzzing.")
    else:
        _valid_corpus = initial_corpus
        logger.info(f"Using all {len(_valid_corpus)} seeds for fuzzing (pass --dry-run to filter).")

    # 6. Initialize & Run Orchestrator
    fuzzer = FusionFuzzLoop(
        config=config,
        strategies=get_strategies(args.project,
                                  stmt_fusion=args.statement_fusion,
                                  dataflow_fusion=args.dataflow_fusion,
                                  all_fusion=args.all_fusion),
        initial_corpus=_valid_corpus,
        all_fusion=args.all_fusion,
    )
    
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
    fuzzer.run(max_iterations=args.iterations, sample_log=sample_log)

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