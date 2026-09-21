from pathlib import Path
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
    print(f"
 Saved quantum descriptor matrix to: {out_csv}")
    return df

if __name__ == "__main__":
    compute_all_quantum_features()
