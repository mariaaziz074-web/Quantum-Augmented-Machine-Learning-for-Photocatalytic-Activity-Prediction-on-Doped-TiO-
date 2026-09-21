# Quantum-Augmented Machine Learning for Photocatalytic Activity Prediction on Doped TiO₂

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![ASE](https://img.shields.io/badge/ASE-Atomic_Simulation_Environment-00599C.svg)](https://wiki.fysik.dtu.dk/ase/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML_Pipeline-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A hybrid quantum-chemical and physics-informed machine learning framework for predicting photocatalytic degradation reaction kinetics and quantum yields across pure and transition-metal/non-metal doped TiO₂ polymorphs (anatase, rutile, P25).


## 🔬 Scientific Overview

Traditional empirical models fail to capture quantum-mechanical electronic effects in doped semiconductors. This project integrates:
1. **Atomistic Structural Modeling**: 13 pristine and doped TiO₂ supercell models (N, C, S, Fe, Cu single-doped and co-doped).
2. **Quantum Feature Extraction**: Semi-empirical and DFT-derived electronic descriptors including HOMO, LUMO, band gap ($\Delta E_g$), Fermi level ($E_F$), ionization potential, and electron affinity.
3. **Hybrid Physics-ML Modeling**: Multi-fidelity XGBoost, Random Forest, and Gradient Boosting regressors mapping quantum descriptors + process parameters ($C_0$, catalyst loading, irradiance, pH) to kinetic rate constants ($k_{obs}$, $\text{min}^{-1}$) and degradation efficiency.
4. **SHAP Feature Interpretability**: Game-theoretic feature attribution identifying band gap narrowing and localized mid-gap impurity states as dominant drivers of visible-light activity.
## 📊 Key Results & Publication Figures

| Figure | Description | File |
| :--- | :--- | :--- |
| **Fig 1** | Quantum band edge alignments (HOMO/LUMO/Band Gap) across 13 catalyst systems | `results/figures/fig1_quantum_band_structures.png` |
| **Fig 2** | Model performance comparison: Empirical vs Hybrid Quantum-ML ($R^2$ / RMSE) | `results/figures/fig2_hybrid_vs_empirical_r2.png` |
| **Fig 3/4**| SHAP game-theoretic global & local feature importance attributions | `results/figures/fig3_shap_kernel_feature_importance_horizontal.png` |
| **Fig 6** | Atomistic 3D rendering panel of all 13 TiO₂ polymorph and doped models | `results/figures/fig6_tio2_structures_panel.png` |

## 📁 Repository Structure

dft-photocatalysis/
├── data/
│   ├── raw/                 # Experimental degradation rate benchmarks
│   └── structures/          # 13 XYZ atomistic structure files for TiO2 systems
├── src/
│   ├── features/            # Quantum descriptor extractors & feature pipelines
│   └── models/              # Hybrid physics-informed ML architectures
├── scripts/
│   ├── 01_fetch_project2_data.py
│   ├── 02_install_xtb_engine.py
│   ├── 08_generate_project3_publication_figures_strong_palette.py
│   ├── 09_generate_project3_shap_figures_kernel_safe.py
│   └── 11_generate_tio2_structures_panel.py
├── results/
│   ├── figures/             # High-resolution (600 DPI) publication figures & PDFs
│   │   └── tio2_structure_renders/  # Individual 3D structure renders
│   └── tables/              # Model benchmark tables & error metrics
├── run_project3_pipeline.py # End-to-end master execution pipeline
├── pyproject.toml           # Project dependencies and packaging metadata
└── README.md
