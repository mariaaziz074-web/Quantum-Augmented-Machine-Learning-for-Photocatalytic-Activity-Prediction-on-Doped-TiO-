from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from xgboost import XGBRegressor

from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

empirical_features = [
    "bandgap_ev", "surface_area_m2g", "catalyst_dosage_gl",
    "initial_dye_conc_mgl", "ph", "light_intensity_mwcm2",
    "reaction_time_min", "temperature_c", "dye_molecular_weight"
]
quantum_features = [
    "dft_homo_ev", "dft_lumo_ev", "dft_bandgap_ev",
    "dft_fermi_ev", "dft_electronegativity_ev",
    "dft_hardness_ev", "dft_dipole_debye"
]
feature_names = empirical_features + quantum_features
target = "degradation_efficiency_percent"

# Neon-ish high-contrast palette (similar vibe to your preference)
COLS = ["#EA1471", "#12767A", "#FF7B00", "#116801CD", "#9B1963",
        "#E72525", "#00A3FF", "#774103F9", "#FF2B55", "#C9097F"]


def catalyst_like_palette(i, top_n):
    return COLS[i % len(COLS)]


def main():
    df = pd.read_csv(DATA_DIR / "hybrid_photocatalysis_dataset.csv").copy()

    missing = [c for c in feature_names + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[feature_names].astype(float).to_numpy()
    y = df[target].astype(float).to_numpy()

    # Train a strong model for explanation
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=800,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)

    # KernelExplainer: choose background + explain samples to keep runtime sane
    rng = np.random.default_rng(42)
    n = X.shape[0]

    bg_size = min(60, n)
    exp_size = min(120, n)

    bg_idx = rng.choice(n, size=bg_size, replace=False)
    exp_idx = rng.choice(n, size=exp_size, replace=False)

    X_bg = X[bg_idx]
    X_explain = X[exp_idx]

    f = model.predict

    print(f"Background size: {bg_size}, explain size: {exp_size}")
    explainer = shap.KernelExplainer(f, X_bg)

    # nsamples controls speed vs quality
    shap_values = explainer.shap_values(X_explain, nsamples=250)

    # shap_values shape: (n_samples, n_features) for single-output regression
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        # sometimes returns list-like structure; take first
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)

    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    out_csv = TABLES_DIR / "project3_shap_kernel_feature_importance_mean_abs.csv"
    imp.to_csv(out_csv, index=False)
    print(f"[OK] Saved table: {out_csv}")

    top_n = min(16, len(feature_names))
    imp_top = imp.head(top_n).iloc[::-1]

    # Horizontal plot
    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=600, facecolor="white")
    cols = [catalyst_like_palette(i, top_n) for i in range(top_n)]
    ax.barh(imp_top["feature"], imp_top["mean_abs_shap"], color=cols, edgecolor="black", linewidth=0.3)

    ax.set_title("SHAP Feature Importance (KernelExplainer; mean |SHAP|)", fontsize=14, pad=14)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))

    plt.tight_layout()
    fig_h = FIGURES_DIR / "fig3_shap_kernel_feature_importance_horizontal.png"
    fig.savefig(fig_h, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Vertical plot
    imp_top2 = imp.head(min(12, len(feature_names)))
    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=600, facecolor="white")
    cols2 = [catalyst_like_palette(i, len(imp_top2)) for i in range(len(imp_top2))]
    ax.bar(imp_top2["feature"], imp_top2["mean_abs_shap"], color=cols2, edgecolor="black", linewidth=0.3)

    ax.set_title("SHAP Feature Importance (Top 12)", fontsize=14, pad=12)
    ax.set_ylabel("Mean |SHAP value|", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    fig_v = FIGURES_DIR / "fig4_shap_kernel_feature_importance_vertical.png"
    fig.savefig(fig_v, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[OK] Saved: {fig_h}")
    print(f"[OK] Saved: {fig_v}")


if __name__ == "__main__":
    main()