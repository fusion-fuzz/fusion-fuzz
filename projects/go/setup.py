import subprocess
import sys
import zipfile
from pathlib import Path


def setup(project_root):
    """
    Sets up the Go fuzzing environment (runs inside the fuzz-go container):
    Go is pre-installed in the golang:latest image — just verify it works,
    then extract seeds.zip.
    """
    print(f"Setting up Go in: {project_root}")

    try:
        result = subprocess.run(
            ["go", "version"], capture_output=True, text=True, check=True
        )
        print(f"Go available: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Error: go not found in PATH: {e}")
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

    print("Go setup complete.")
