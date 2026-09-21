from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "results" / "figures"
TABLES_DIR = ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


# Match Project 3 feature definitions
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


# Neon-ish high contrast palette
NEON = [
    "#FF006E", "#00F5FF", "#F9FF00", "#39FF14", "#FF3D00",
    "#7C00FF", "#00A3FF", "#FF8A00", "#FF2BF2", "#A7FF00"
]


def catalyst_palette_for_rank(rank, n_bars):
    # rotate through neon colors deterministically
    return NEON[rank % len(NEON)]


def main():
    df = pd.read_csv(DATA_DIR / "hybrid_photocatalysis_dataset.csv")

    missing = [c for c in feature_names + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in hybrid dataset: {missing}")

    X = df[feature_names].astype(float).to_numpy()
    y = df[target].astype(float).to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Strong model for SHAP (XGBoost)
    # (Pipeline includes scaling, but tree models don't need it; keep minimal)
   xgb_params = dict(
    objective="reg:squarederror",
    base_score=0.5,          # <-- IMPORTANT: stops SHAP base_score parsing error
    n_estimators=450,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

    # Accumulate mean |SHAP| per feature across folds
    shap_abs_sum = np.zeros(len(feature_names), dtype=float)
    r2s = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X), start=1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = XGBRegressor(**xgb_params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        r2 = r2_score(y_te, preds)
        r2s.append(r2)

        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_te)

        # shap can return list; normalize to array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[0]

        shap_vals = np.asarray(shap_vals)  # (n_samples, n_features)
        shap_abs_sum += np.abs(shap_vals).mean(axis=0)

        print(f"Fold {fold}/5 done | R2={r2:.4f}")

    shap_abs_mean = shap_abs_sum / kf.get_n_splits

    imp = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": shap_abs_mean
    }).sort_values("mean_abs_shap", ascending=False)

    out_csv = TABLES_DIR / "project3_shap_feature_importance_mean_abs.csv"
    imp.to_csv(out_csv, index=False)

    # Plot: horizontal bar (top 16)
    top_n = min(16, len(feature_names))
    imp_top = imp.head(top_n).iloc[::-1]  # reverse for barh

    fig, ax = plt.subplots(figsize=(12.5, 7.2), dpi=600, facecolor="white")
    ax.set_facecolor("white")

    colors = [catalyst_palette_for_rank(i, top_n) for i in range(top_n)]
    ax.barh(imp_top["feature"], imp_top["mean_abs_shap"], color=colors, edgecolor="black", linewidth=0.3)

    ax.set_title("SHAP Feature Importance (Hybrid ML; mean |SHAP| across 5 folds)",
                 fontsize=14, pad=14)
    ax.set_xlabel("Mean |SHAP value|", fontsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()

    fig1_png = FIGURES_DIR / "fig3_shap_feature_importance_horizontal.png"
    fig1_pdf = FIGURES_DIR / "fig3_shap_feature_importance_horizontal.pdf"
    fig.savefig(fig1_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(fig1_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # Plot: vertical bar (top 12)
    top_n2 = min(12, len(feature_names))
    imp_top2 = imp.head(top_n2)

    fig, ax = plt.subplots(figsize=(12.5, 5.8), dpi=600, facecolor="white")
    ax.set_facecolor("white")

    colors2 = [catalyst_palette_for_rank(i, top_n2) for i in range(top_n2)]
    ax.bar(imp_top2["feature"], imp_top2["mean_abs_shap"], color=colors2, edgecolor="black", linewidth=0.3)

    ax.set_title("SHAP Feature Importance (Top 12)", fontsize=14, pad=12)
    ax.set_ylabel("Mean |SHAP value|", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    plt.xticks(rotation=35, ha="right")

    plt.tight_layout()
    fig2_png = FIGURES_DIR / "fig4_shap_feature_importance_vertical.png"
    fig2_pdf = FIGURES_DIR / "fig4_shap_feature_importance_vertical.pdf"
    fig.savefig(fig2_png, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(fig2_pdf, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print("\nDone SHAP figures.")
    print(f"R2 across folds (XGB hybrid): mean={np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
    print(f"Saved: {out_csv}")
    print(f"Saved: {fig1_png}")
    print(f"Saved: {fig2_png}")


if __name__ == "__main__":
    main()