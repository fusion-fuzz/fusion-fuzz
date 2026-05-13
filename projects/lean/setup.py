import os
import sys
import shutil
import subprocess


def setup(project_root):
    """
    Sets up the Lean 4 fuzzing environment (runs inside the ffl-lean container):
    1. Verifies lean is available via elan.
    2. Clones leanprover/lean4 to harvest seed .lean files.
    3. Copies lean4 test seed files into projects/lean/seeds/.
    4. Downloads internlm/Lean-Workbook dataset into corpus.db if venv available.
    """
    print(f"Setting up Lean 4 in: {project_root}")

    lean_bin = "/home/ffluser/.elan/bin/lean"

    # 1. Verify lean
    try:
        result = subprocess.run(
            [lean_bin, "--version"], capture_output=True, text=True, check=True
        )
        print(f"Lean available: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: lean not found at {lean_bin}: {e}")
        sys.exit(1)

    # 2. Clone lean4 for seed harvesting
    lean4_dir = os.path.join(project_root, "lean4")
    if not os.path.exists(lean4_dir):
        print("Cloning leanprover/lean4 repository (for test seeds)...")
        try:
            subprocess.run(
                ["git", "clone", "--depth=1", "https://github.com/leanprover/lean4.git"],
                cwd=project_root, check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Clone failed: {e}")
            sys.exit(1)
    else:
        print("lean4 repository already exists.")

    # 3. Collect .lean seed files
    seeds_dir = os.path.join(project_root, "seeds")
    os.makedirs(seeds_dir, exist_ok=True)

    seed_source_dirs = [
        os.path.join(lean4_dir, "tests", "lean"),
        os.path.join(lean4_dir, "tests", "compiler"),
        os.path.join(lean4_dir, "tests", "elaboration"),
        os.path.join(lean4_dir, "tests", "structure"),
        os.path.join(lean4_dir, "tests", "tactics"),
    ]

    collected = skipped = 0
    for source_dir in seed_source_dirs:
        if not os.path.exists(source_dir):
            continue
        for root, _, files in os.walk(source_dir):
            for fname in files:
                if not fname.endswith(".lean"):
                    continue
                src_path = os.path.join(root, fname)
                rel = os.path.relpath(src_path, lean4_dir)
                safe_name = rel.replace(os.sep, "__")
                dst_path = os.path.join(seeds_dir, safe_name)
                if os.path.exists(dst_path):
                    skipped += 1
                    continue
                try:
                    shutil.copy2(src_path, dst_path)
                    collected += 1
                except Exception as e:
                    print(f"Warning: could not copy {src_path}: {e}")

    print(f"Collected {collected} new .lean seeds ({skipped} already existed).")

    # 4. Download internlm/Lean-Workbook dataset
    ffl_root = os.path.dirname(os.path.dirname(project_root))
    venv_python = os.path.join(ffl_root, "openai-env", "bin", "python")
    dataset_script = os.path.join(project_root, "dataset.py")

    if not os.path.exists(venv_python):
        print(
            f"Warning: openai-env not found at {venv_python}. "
            "Skipping Lean-Workbook download.\n"
            "Activate the venv and run  python projects/lean/dataset.py  manually."
        )
    elif not os.path.exists(dataset_script):
        print(f"Warning: {dataset_script} not found — skipping dataset download.")
    else:
        print("Downloading internlm/Lean-Workbook dataset...")
        try:
            subprocess.run([venv_python, dataset_script], check=True, cwd=project_root)
            print("Lean-Workbook dataset download complete.")
        except subprocess.CalledProcessError as e:
            print(f"Dataset download failed (non-fatal): {e}")

    print("Lean 4 setup complete.")
