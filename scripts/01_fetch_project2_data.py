from pathlib import Path
import shutil
import json
import subprocess

REPO_URL = "https://github.com/mariaaziz074-web/photocatalysis-research-benchmark"
REPO_NAME = "photocatalysis-research-benchmark"
TAG_OR_REF = "v1.0.0"

FILES_TO_COPY = [
    "results/tables/benchmark_results.csv",
    "results/tables/catalyst_distribution.csv",
    "results/tables/dye_distribution.csv",
    "results/tables/dataset_summary_statistics.csv",
]

ROOT = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = ROOT / "data" / "external"
RAW_DIR = ROOT / "data" / "raw"
CLONE_DIR = EXTERNAL_DIR / REPO_NAME

RAW_DIR.mkdir(parents=True, exist_ok=True)
EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)

def ensure_clone():
    if not CLONE_DIR.exists():
        print(f"Cloning {REPO_URL} -> {CLONE_DIR}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(CLONE_DIR)], check=True)

def copy_files():
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(CLONE_DIR)).decode().strip()
    manifest = {"source_repo": REPO_URL, "commit": commit_hash, "copied": []}

    for rel_path in FILES_TO_COPY:
        src = CLONE_DIR / rel_path
        if src.exists():
            dst = RAW_DIR / Path(rel_path).name
            shutil.copy2(src, dst)
            manifest["copied"].append(str(dst))

    manifest_path = RAW_DIR / "project2_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f" Successfully imported Project 2 data into {RAW_DIR}")

if __name__ == "__main__":
    ensure_clone()
    copy_files()
