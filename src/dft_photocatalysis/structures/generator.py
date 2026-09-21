from pathlib import Path
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
