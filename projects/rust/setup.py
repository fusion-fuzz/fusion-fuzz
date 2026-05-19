import os
import shutil
import subprocess
import zipfile


def setup(project_root):
    """
    Sets up the Rust fuzzing environment (runs inside the fuzz-rust docker):
    1. Verifies rustc is available on PATH.
    2. Extracts seeds.zip as the initial corpus if not already present.
    """
    print(f"Setting up Rust in: {project_root}")

    # 1. Check rustc exists
    rustc_path = shutil.which("rustc")
    if not rustc_path:
        raise RuntimeError("rustc not found on PATH — run inside the fuzz-rust docker container")
    result = subprocess.run([rustc_path, "--version"], capture_output=True, text=True)
    print(f"Found rustc: {result.stdout.strip()}")

    # 2. Extract seeds.zip to project_root/seeds/ if not already done
    seeds_dir = os.path.join(project_root, "seeds")
    seeds_zip = os.path.join(project_root, "seeds.zip")

    if os.path.isdir(seeds_dir) and os.listdir(seeds_dir):
        print(f"Seeds directory already populated at {seeds_dir}, skipping extraction.")
    else:
        if not os.path.exists(seeds_zip):
            raise FileNotFoundError(f"seeds.zip not found at {seeds_zip}")
        print(f"Extracting {seeds_zip} → {project_root} ...")
        with zipfile.ZipFile(seeds_zip, "r") as zf:
            zf.extractall(project_root)
        count = len([f for f in os.listdir(seeds_dir) if f.endswith(".rs")])
        print(f"Extracted {count} .rs seed files to {seeds_dir}")

    print("Rust setup complete.")
