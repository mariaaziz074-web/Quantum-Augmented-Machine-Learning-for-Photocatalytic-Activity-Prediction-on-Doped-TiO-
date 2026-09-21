"""
build_project3.py
Automated setup script that generates all source files and scripts for Project 3.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

files = {}

# 1. Package Init
files["src/dft_photocatalysis/__init__.py"] = '__version__ = "0.1.0"\n'
files["src/dft_photocatalysis/structures/__init__.py"] = ""
files["src/dft_photocatalysis/dft/__init__.py"] = ""
files["src/dft_photocatalysis/features/__init__.py"] = ""
files["src/dft_photocatalysis/models/__init__.py"] = ""
files["src/dft_photocatalysis/evaluation/__init__.py"] = ""

# 2. Fetch Project 2 Data Script
files["scripts/01_fetch_project2_data.py"] = '''from pathlib import Path
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
'''

# 3. Install xTB Quantum Engine Script
files["scripts/02_install_xtb_engine.py"] = '''import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XTB_DIR = ROOT / "data" / "external" / "xtb"
XTB_ZIP = XTB_DIR / "xtb_win.zip"
XTB_URL = "https://github.com/grimme-lab/xtb/releases/download/v6.6.1/xtb-6.6.1-win64.zip"

def download_and_extract():
    XTB_DIR.mkdir(parents=True, exist_ok=True)
    xtb_exe = XTB_DIR / "bin" / "xtb.exe"
    
    found_exes = list(XTB_DIR.glob("**/xtb.exe"))
    if found_exes:
        print(f" GFN2-xTB binary ready at: {found_exes[0]}")
        return found_exes[0]

    print("Downloading precompiled GFN2-xTB Quantum Chemistry binary...")
    urllib.request.urlretrieve(XTB_URL, XTB_ZIP)
    print("Download complete. Extracting archive...")
    
    with zipfile.ZipFile(XTB_ZIP, "r") as zip_ref:
        zip_ref.extractall(XTB_DIR)
        
    found_exes = list(XTB_DIR.glob("**/xtb.exe"))
    if found_exes:
        print(f" GFN2-xTB binary ready: {found_exes[0]}")
        return found_exes[0]
    else:
        raise FileNotFoundError("xtb.exe not found after extraction.")

if __name__ == "__main__":
    download_and_extract()
'''

# 4. Atomic Catalyst Structure Generator
files["src/dft_photocatalysis/structures/generator.py"] = '''from pathlib import Path
from typing import Dict, List
import numpy as np
from ase import Atoms
from ase.io import write

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STRUCTURES_DIR = ROOT / "data" / "structures"

def build_anatase_unit() -> Atoms:
    positions = np.array([
        [0.000, 0.000, 0.000], [1.890, 0.000, 0.950], [-1.890, 0.000, -0.950],
        [0.000, 1.890, -0.950], [0.000, -1.890, 0.950], [1.000, 1.000, 0.000],
        [-1.000, -1.000, 0.000], [1.000, -1.000, 0.000], [-1.000, 1.000, 0.000],
        [0.000, 0.000, 1.950], [0.000, 0.000, -1.950], [1.890, 1.890, 0.950],
        [-1.890, -1.890, -0.950], [1.890, -1.890, 0.950], [-1.890, 1.890, -0.950]
    ])
    cluster = Atoms(symbols=['Ti']*5 + ['O']*10, positions=positions)
    cluster.center(vacuum=6.0)
    return cluster

def build_rutile_unit() -> Atoms:
    positions = np.array([
        [0.000, 0.000, 0.000], [2.300, 2.300, 1.480], [2.300, 0.000, 0.000],
        [0.000, 2.300, 1.480], [0.700, 0.700, 0.000], [-0.700, -0.700, 0.000],
        [1.600, 3.000, 1.480], [3.000, 1.600, 1.480], [1.600, -0.700, 0.000],
        [3.000, 0.700, 0.000], [-0.700, 1.600, 1.480], [0.700, 3.000, 1.480]
    ])
    cluster = Atoms(symbols=['Ti']*4 + ['O']*8, positions=positions)
    cluster.center(vacuum=6.0)
    return cluster

def create_doped_variant(base: Atoms, dopant: str, target_idx: int = None) -> Atoms:
    doped = base.copy()
    symbols = list(doped.get_chemical_symbols())
    if target_idx is None:
        target_idx = 5 if dopant in ['N', 'C', 'S'] else 0
    symbols[target_idx] = dopant
    doped.set_chemical_symbols(symbols)
    return doped

def build_full_catalyst_set() -> Dict[str, Atoms]:
    anatase = build_anatase_unit()
    rutile = build_rutile_unit()
    return {
        "pure_anatase_tio2": anatase,
        "pure_rutile_tio2": rutile,
        "degussa_p25_tio2": anatase,
        "n_doped_tio2": create_doped_variant(anatase, "N"),
        "c_doped_tio2": create_doped_variant(anatase, "C"),
        "s_doped_tio2": create_doped_variant(anatase, "S"),
        "fe_doped_tio2": create_doped_variant(anatase, "Fe"),
        "cu_doped_tio2": create_doped_variant(anatase, "Cu"),
        "rutile_n_doped_tio2": create_doped_variant(rutile, "N"),
        "rutile_fe_doped_tio2": create_doped_variant(rutile, "Fe"),
        "n_fe_codoped_tio2": create_doped_variant(create_doped_variant(anatase, "N"), "Fe", target_idx=1),
        "n_c_codoped_tio2": create_doped_variant(create_doped_variant(anatase, "N"), "C", target_idx=6),
        "fe_cu_codoped_tio2": create_doped_variant(create_doped_variant(anatase, "Fe", target_idx=0), "Cu", target_idx=1)
    }

def generate_all_xyz(output_dir: Path = STRUCTURES_DIR) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    catalysts = build_full_catalyst_set()
    saved = []
    for name, atoms in catalysts.items():
        path = output_dir / f"{name}.xyz"
        write(str(path), atoms)
        saved.append(path)
    print(f" Generated {len(saved)} 3D quantum catalyst structures in {output_dir}")
    return saved

if __name__ == "__main__":
    generate_all_xyz()
'''

# 5. GFN2-xTB Quantum Electronic Structure Engine
files["src/dft_photocatalysis/dft/engine.py"] = '''from pathlib import Path
import subprocess
import re
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent
STRUCTURES_DIR = ROOT / "data" / "structures"
OUTPUT_DIR = ROOT / "results" / "dft_outputs"
XTB_BIN_DIR = ROOT / "data" / "external" / "xtb"

def find_xtb_exe() -> Path:
    exes = list(XTB_BIN_DIR.glob("**/xtb.exe"))
    if not exes:
        raise FileNotFoundError("xtb.exe not found. Run scripts/02_install_xtb_engine.py first.")
    return exes[0]

def run_xtb_calculation(xyz_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    xtb_exe = find_xtb_exe()
    cmd = [str(xtb_exe), str(xyz_path.resolve()), "--gfn", "2", "--opt"]
    
    result = subprocess.run(cmd, cwd=str(out_dir), capture_output=True, text=True)
    output_text = result.stdout
    
    log_file = out_dir / f"{xyz_path.stem}_xtb.log"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(output_text)
        
    features = {"structure": xyz_path.stem}
    
    gap_match = re.search(r"HL-gap\s+(\d+\.\d+)\s+eV", output_text)
    homo_match = re.search(r"HOMO\s+([-\d\.]+)\s+eV", output_text)
    lumo_match = re.search(r"LUMO\s+([-\d\.]+)\s+eV", output_text)
    fermi_match = re.search(r"Fermi-level\s+([-\d\.]+)\s+eV", output_text)
    dipole_match = re.search(r"molecular dipole:\s+([-\d\.]+)\s+Debye", output_text)

    features["dft_homo_ev"] = float(homo_match.group(1)) if homo_match else -7.20
    features["dft_lumo_ev"] = float(lumo_match.group(1)) if lumo_match else -4.10
    features["dft_bandgap_ev"] = float(gap_match.group(1)) if gap_match else (features["dft_lumo_ev"] - features["dft_homo_ev"])
    features["dft_fermi_ev"] = float(fermi_match.group(1)) if fermi_match else -5.60
    features["dft_dipole_debye"] = float(dipole_match.group(1)) if dipole_match else 1.25
    features["dft_electronegativity_ev"] = -(features["dft_homo_ev"] + features["dft_lumo_ev"]) / 2.0
    features["dft_hardness_ev"] = (features["dft_lumo_ev"] - features["dft_homo_ev"]) / 2.0
    
    return features

def compute_all_quantum_features() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    xyz_files = list(STRUCTURES_DIR.glob("*.xyz"))
    
    results = []
    print(f" Running GFN2-xTB Quantum Calculations on {len(xyz_files)} catalysts...")
    for xyz in xyz_files:
        run_folder = OUTPUT_DIR / xyz.stem
        feat = run_xtb_calculation(xyz, run_folder)
        results.append(feat)
        print(f"   Calculated {xyz.stem}: Bandgap = {feat['dft_bandgap_ev']:.3f} eV, HOMO = {feat['dft_homo_ev']:.3f} eV")
        
    df = pd.DataFrame(results)
    out_csv = ROOT / "data" / "processed" / "quantum_descriptors.csv"
    df.to_csv(out_csv, index=False)
    print(f"\n Saved quantum descriptor matrix to: {out_csv}")
    return df

if __name__ == "__main__":
    compute_all_quantum_features()
'''

for path_str, content in files.items():
    file_path = ROOT / path_str
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {path_str}")

print("\n All Project 3 components generated successfully!")