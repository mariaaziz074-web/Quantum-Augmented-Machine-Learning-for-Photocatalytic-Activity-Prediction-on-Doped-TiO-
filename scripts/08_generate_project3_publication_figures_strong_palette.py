from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"


def catalyst_label(key: str) -> str:
    k = str(key).lower()
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
        "n_fe_codoped_tio2": "N-Fe co-doped",
        "n_c_codoped_tio2": "N-C co-doped",
        "fe_cu_codoped_tio2": "Fe-Cu co-doped",
    }
    return mapping.get(k, str(key))


def generate_fig1_strong():
    q_csv = PROCESSED_DIR / "quantum_descriptors.csv"
    df_q = pd.read_csv(q_csv).copy()

    required = {"catalyst_key", "dft_homo_ev", "dft_lumo_ev", "dft_bandgap_ev"}
    if not required.issubset(df_q.columns):
        raise ValueError(f"quantum_descriptors.csv missing columns. Required={required}")

    df_q = df_q.sort_values("dft_bandgap_ev", ascending=False).reset_index(drop=True)
    n = len(df_q)
    labels = [catalyst_label(x) for x in df_q["catalyst_key"].tolist()]

    homo = df_q["dft_homo_ev"].astype(float).to_numpy()
    lumo = df_q["dft_lumo_ev"].astype(float).to_numpy()
    gaps = df_q["dft_bandgap_ev"].astype(float).to_numpy()
    y = np.arange(n)

    # Dark-bright high-contrast palette
    homo_col = "#130688"   # deep navy
    lumo_col = "#ED2914"   # bright orange-red
    gap_cols = [
    "#FF006E",  # neon pink
    "#6D035A",  # electric cyan
    "#BB260F",  # electric yellow
    "#23E69E",  # neon green
    "#FF3D00",  # neon orange-red
    "#4B0497",  # vivid violet
    "#00A3FF",  # electric blue
    "#DD0F0F",  # bright amber
    "#CB10BF",  # hot fuchsia
    "#016C1D"   # chartreuse
]

    grid_col = "#E6EAF0"
    axis_col = "#111827"

    energy_min = float(min(homo.min(), lumo.min()))
    energy_max = float(max(homo.max(), lumo.max()))
    pad = max(0.35, 0.07 * (energy_max - energy_min))
    xlim = (energy_min - pad, energy_max + pad)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=600, facecolor="white")
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.6, 7.2, 2.2], wspace=0.08)

    ax_lab = fig.add_subplot(gs[0, 0])
    ax_band = fig.add_subplot(gs[0, 1])
    ax_gap = fig.add_subplot(gs[0, 2], sharey=ax_band)

    # Left label column
    ax_lab.axis("off")
    ax_lab.set_xlim(0, 1)
    ax_lab.set_ylim(-0.5, n - 0.5)
    for i in range(n):
        ax_lab.text(0.02, i, labels[i], ha="left", va="center",
                    fontsize=10.2, color=axis_col)

    # Band panel
    ax_band.set_xlim(*xlim)
    ax_band.set_ylim(-0.5, n - 0.5)
    ax_band.grid(True, axis="x", linestyle="--", alpha=0.35, color=grid_col)
    ax_band.grid(False, axis="y")

    for i in range(n):
        # Gap line = connecting line
        ax_band.plot([homo[i], lumo[i]], [y[i], y[i]],
                      color=gap_cols[i % len(gap_cols)], linewidth=3.0, alpha=1.0, zorder=2)
        ax_band.scatter([homo[i]], [y[i]], s=48, color=homo_col,
                        edgecolor="white", linewidth=0.7, zorder=3)
        ax_band.scatter([lumo[i]], [y[i]], s=48, color=lumo_col,
                        edgecolor="white", linewidth=0.7, marker="D", zorder=3)

    ax_band.set_yticks(y)
    ax_band.set_yticklabels([])
    ax_band.tick_params(axis="y", left=False, labelleft=False)

    ax_band.set_xlabel("Energy level vs. vacuum (eV)",
                        fontsize=11, fontweight="bold", color=axis_col)
    ax_band.set_title("Quantum electronic structure: band alignment & frontier levels",
                      fontsize=13, pad=12, color=axis_col)

    legend_handles = [
        Line2D([0], [0], color=homo_col, marker="o", linestyle="None", markersize=7, label="HOMO"),
        Line2D([0], [0], color=lumo_col, marker="D", linestyle="None", markersize=7, label="LUMO"),
        Line2D([0], [0], color=gap_cols[0], linestyle="-", linewidth=3.0, label="Gap (HOMO->LUMO)"),
    ]
    ax_band.legend(handles=legend_handles, loc="upper left",
                   frameon=True, facecolor="white", framealpha=1.0,
                   edgecolor="#D1D5DB", fontsize=9.8, borderpad=0.6)

    # Gap panel
    ax_gap.barh(
    y, gaps, height=0.62,
    color=[gap_cols[i % len(gap_cols)] for i in range(n)],
    alpha=1.0, edgecolor="none"
)
    ax_gap.grid(True, axis="x", linestyle="--", alpha=0.25, color=grid_col)
    ax_gap.set_yticks(y)
    ax_gap.set_yticklabels([])
    ax_gap.tick_params(axis="y", left=False, labelleft=False)

    ax_gap.set_xlabel("Bandgap $E_g$ (eV)",
                      fontsize=11, fontweight="bold", color=axis_col)

    fig.subplots_adjust(left=0.03, right=0.995, top=0.90, bottom=0.14)

    out_png = FIGURES_DIR / "fig1_quantum_band_structures.png"
    out_pdf = FIGURES_DIR / "fig1_quantum_band_structures.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def generate_fig2_strong():
    bm_csv = TABLES_DIR / "hybrid_vs_empirical_benchmark.csv"
    df_bm = pd.read_csv(bm_csv).copy()

    required = {"Model", "Empirical_R2", "Hybrid_R2"}
    if not required.issubset(df_bm.columns):
        raise ValueError(f"hybrid_vs_empirical_benchmark.csv missing columns. Required={required}")

    models = df_bm["Model"].astype(str).tolist()
    emp = df_bm["Empirical_R2"].astype(float).to_numpy()
    hyb = df_bm["Hybrid_R2"].astype(float).to_numpy()

    x = np.arange(len(models))
    w = 0.42

    # Dark-bright high-contrast palette
    emp_col = "#003366"
    hyb_col = "#E00808"

    grid_col = "#E6EAF0"
    axis_col = "#111827"

    fig, ax = plt.subplots(figsize=(10.2, 5.6), dpi=600, facecolor="white")
    ax.set_facecolor("white")

    bars_emp = ax.bar(x - w/2, emp, width=w, color=emp_col, alpha=1.0, edgecolor="none",
                       label="Pure Empirical ML")
    bars_hyb = ax.bar(x + w/2, hyb, width=w, color=hyb_col, alpha=1.0, edgecolor="none",
                       label="Quantum-Hybrid ML (Ours)")

    ax.set_title("Model performance: pure empirical vs quantum-hybrid features",
                 fontsize=13, pad=10, color=axis_col, fontweight="bold")
    ax.set_ylabel("Test R2 score", fontsize=11, fontweight="bold", color=axis_col)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, fontweight="bold", color=axis_col)

    ax.grid(True, axis="y", linestyle="--", alpha=0.25, color=grid_col)
    y_max = float(max(emp.max(), hyb.max()))
    ax.set_ylim(0, min(1.08, y_max + 0.10))

    ax.legend(frameon=True, facecolor="white", edgecolor="#D1D5DB",
              framealpha=1.0, fontsize=9.8, loc="upper right")

    for bars in (bars_emp, bars_hyb):
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.012, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold",
                    color=axis_col, clip_on=True)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.84, bottom=0.22)

    out_png = FIGURES_DIR / "fig2_hybrid_vs_empirical_r2.png"
    out_pdf = FIGURES_DIR / "fig2_hybrid_vs_empirical_r2.pdf"
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"[OK] Saved {out_png}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    generate_fig1_strong()
    generate_fig2_strong()


if __name__ == "__main__":
    main()

