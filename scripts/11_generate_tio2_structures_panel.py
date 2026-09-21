from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
STRUCTURES_DIR = ROOT / "data" / "structures"
FIGURES_DIR = ROOT / "results" / "figures"
INDIV_DIR = FIGURES_DIR / "tio2_structure_renders"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
INDIV_DIR.mkdir(parents=True, exist_ok=True)

# Desired order (match your model naming convention)
ORDER = [
    "pure_anatase_tio2",
    "degussa_p25_tio2",
    "pure_rutile_tio2",
    "n_doped_tio2",
    "c_doped_tio2",
    "s_doped_tio2",
    "fe_doped_tio2",
    "cu_doped_tio2",
    "rutile_n_doped_tio2",
    "rutile_fe_doped_tio2",
    "n_fe_codoped_tio2",
    "n_c_codoped_tio2",
    "fe_cu_codoped_tio2",
]

def label_for(key: str) -> str:
    k = key.lower()
    mapping = {
        "degussa_p25_tio2": "Degussa P25",
        "pure_anatase_tio2": "Anatase",
        "pure_rutile_tio2": "Rutile",
        "n_doped_tio2": "N-doped",
        "c_doped_tio2": "C-doped",
        "s_doped_tio2": "S-doped",
        "fe_doped_tio2": "Fe-doped",
        "cu_doped_tio2": "Cu-doped",
        "rutile_n_doped_tio2": "Rutile N-doped",
        "rutile_fe_doped_tio2": "Rutile Fe-doped",
        "n_fe_codoped_tio2": "N–Fe co-doped",
        "n_c_codoped_tio2": "N–C co-doped",
        "fe_cu_codoped_tio2": "Fe–Cu co-doped",
    }
    return mapping.get(k, key)

# High-contrast element colors (neon-like but still clean on white)
ELEMENT_COLORS = {
    "Ti": "#0CA553",   # deep navy
    "O":  "#FF2D2D",   # bright red
    "N":  "#00CCFF",   # cyan
    "C":  "#2B00FF",   # electric yellow
    "S":  "#39FF14",   # neon green
    "Fe": "#890556F5",   # vivid violet
    "Cu": "#FF8A00",   # bright amber/orange
}

# Atom sizes for visuals
ELEMENT_SIZES = {
    "Ti": 70, "O": 40, "N": 55, "C": 45, "S": 55, "Fe": 60, "Cu": 60
}

def render_atoms_3d(ax, atoms, title: str):
    pos = atoms.get_positions().astype(float)
    syms = atoms.get_chemical_symbols()

    # Center camera on dopant atoms (or fallback to overall center)
    dopant_idx = [i for i, s in enumerate(syms) if s not in ["Ti", "O"]]
    if len(dopant_idx) > 0:
        center = pos[dopant_idx].mean(axis=0)
    else:
        center = pos.mean(axis=0)

    pos = pos - center

    # Normalize size so camera distance is consistent
    r = np.linalg.norm(pos, axis=1).max()
    if r > 0:
        pos = pos / r

    # 3D scatter
    for elem in sorted(set(syms)):
        idx = [i for i, s in enumerate(syms) if s == elem]
        p = pos[idx]
        sizes = ELEMENT_SIZES.get(elem, 45)
        color = ELEMENT_COLORS.get(elem, "#B20863")
        ax.scatter(
            p[:, 0], p[:, 1], p[:, 2],
            s=sizes, c=color,
            depthshade=False,
            edgecolors="white",
            linewidths=0.6
        )

    ax.set_title(title, fontsize=9.8, pad=6)

    # Clean publication look
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.grid(False)
    
    # Royal blue pane/background tint (sharp, not neon)
    pane_rgb = (47/255, 128/255, 237/255)  # #2F80ED
    pane_alpha = 0.28  # slightly higher than subtle so it is visible
    pane_rgba = (pane_rgb[0], pane_rgb[1], pane_rgb[2], pane_alpha)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.set_pane_color(pane_rgba)
        except Exception:
            pass
    # Camera angle (more visible dopant)
    ax.view_init(elev=25, azim=95)

    # Equal aspect-ish bounds
    lim = 1.05
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
def main():
    # Gather existing xyz files
    missing = []
    xyz_paths = {}
    for key in ORDER:
        p = STRUCTURES_DIR / f"{key}.xyz"
        if p.exists():
            xyz_paths[key] = p
        else:
            missing.append(key)

    if missing:
        raise FileNotFoundError(f"Missing XYZ structure files for: {missing}")

    # Render individual images too
    for key in ORDER:
        atoms = read(str(xyz_paths[key]))
        fig = plt.figure(figsize=(5.0, 4.3), dpi=600)
        ax = fig.add_subplot(111, projection="3d")
        render_atoms_3d(ax, atoms, label_for(key))
        out = INDIV_DIR / f"{key}.png"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    # Panel render (3 rows x 5 cols = 15 slots, we use first 13)
    n = len(ORDER)
    nrows, ncols = 3, 5
    fig = plt.figure(figsize=(18.5, 11.0), dpi=600, facecolor="white")

    for i, key in enumerate(ORDER):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        atoms = read(str(xyz_paths[key]))
        render_atoms_3d(ax, atoms, label_for(key))

    # Hide unused panels if any
    for j in range(n, nrows * ncols):
        ax_unused = fig.add_subplot(nrows, ncols, j + 1, projection="3d")
        ax_unused.axis("off")

    fig.suptitle("TiO$_2$ polymorphs and doped catalysts (atomistic structure models)",
                 fontsize=14, y=0.995)

    panel_png = FIGURES_DIR / "fig6_tio2_structures_panel.png"
    panel_pdf = FIGURES_DIR / "fig6_tio2_structures_panel.pdf"
    fig.savefig(panel_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(panel_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Panel saved: {panel_png}")
    print(f"[OK] Individual renders saved in: {INDIV_DIR}")

if __name__ == "__main__":
    main()