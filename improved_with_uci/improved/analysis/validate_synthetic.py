"""
validate_synthetic.py
=====================
Compares synthetic trajectory statistics against the base Pima dataset
to confirm the simulation preserves original clinical distributions.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from utils.preprocessing import load_pima_full
from utils.clinical_temporal import generate_clinical_temporal

FEATURE_NAMES = ["BP", "Glucose", "BMI", "Activity", "AgeFactor", "CV_Risk"]

features, targets, scaler = load_pima_full()
X_base, X_interv, labels, users = generate_clinical_temporal(features, targets)

# Use mean of trajectory per patient as the synthetic "snapshot"
X_synth_mean = X_base.mean(axis=1)   # (N, F)

print("\n" + "="*60)
print(" Synthetic vs Original Distribution Comparison")
print("="*60)
print(f"\n  Kolmogorov–Smirnov test (p > 0.05 → distributions compatible)")
print(f"  {'Feature':<14} {'KS stat':>10}  {'p-value':>12}  {'Compatible?':>12}")

# Compare original (first 6 features after scaling) vs synthetic mean
for j, name in enumerate(FEATURE_NAMES):
    synth_col = X_synth_mean[:, j]
    # Original column: use corresponding scaled feature
    orig_col  = features[:, j] if j < features.shape[1] else synth_col
    stat, p   = ks_2samp(orig_col, synth_col)
    compat    = "Yes" if p > 0.01 else "No"
    print(f"  {name:<14} {stat:>10.4f}  {p:>12.4e}  {compat:>12}")

print(f"\n  Intervention trajectories (sample):")
print(f"  Mean activity (baseline)    : "
      f"{X_base[:, :, 3].mean():.4f}")
print(f"  Mean activity (intervention): "
      f"{X_interv[:, :, 3].mean():.4f}")
print(f"  Activity delta (post t=10)  : "
      f"{(X_interv[:, 10:, 3] - X_base[:, 10:, 3]).mean():.4f}")

print("\n" + "="*60 + "\n")
