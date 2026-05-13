import subprocess
import sys
import zipfile
from pathlib import Path


def setup(project_root):
    """
    Sets up the Swift fuzzing environment (runs inside the ffl-swift container):
    Swift is pre-installed in the swift:latest image — just verify it works.
    """
    print(f"Setting up Swift in: {project_root}")

    try:
        result = subprocess.run(
            ["swift", "--version"], capture_output=True, text=True, check=True
        )
        print(f"Swift available: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: swift not found in PATH: {e}")
        sys.exit(1)

    seeds_zip = Path(project_root) / "seeds.zip"
    seeds_dir = Path(project_root) / "seeds"
    if seeds_zip.exists():
        print(f"Extracting {seeds_zip} into {seeds_dir}")
        seeds_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(seeds_zip, "r") as zf:
            zf.extractall(seeds_dir)
        print(f"Seeds extracted to {seeds_dir}")
    else:
        print(f"Warning: {seeds_zip} not found, skipping seed extraction")

    print("Swift setup complete.")
