"""
realism_validation.py
=====================
Statistical validation of synthetic trajectories against clinical ranges.

Checks:
  1. Descriptive stats (mean, std, min, max) per feature
  2. Correlation with outcome (should be moderate, not deterministic)
  3. Normality test per feature (Shapiro-Wilk on sample)
  4. Prints a pass/fail table against known clinical reference ranges
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy.stats import shapiro, pearsonr
from utils.preprocessing import load_pima_full
from utils.clinical_temporal import generate_clinical_temporal

# ── Clinical reference ranges (normalised values are approximate) ──────────
# These are approximate z-score ranges for Pima diabetic population
REF_RANGES = {
    "BP":        (-3.0, 3.0),
    "Glucose":   (-3.0, 3.0),
    "BMI":       (-3.0, 3.0),
    "Activity":  (0.0,  1.0),
    "AgeFactor": (0.0,  0.3),
    "CV_Risk":   (-3.0, 3.0),
}

FEATURE_NAMES = list(REF_RANGES.keys())

print("\n" + "="*60)
print(" Synthetic Trajectory Realism Validation")
print("="*60)

features, targets, scaler = load_pima_full()
X_base, _, labels, users  = generate_clinical_temporal(features, targets)

N, T, F = X_base.shape
print(f"\n  Dataset: {N} patients × {T} time steps × {F} features")
print(f"  Label prevalence (7-day): {labels[:,0].mean()*100:.1f}%  "
      f"(30-day: {labels[:,1].mean()*100:.1f}%  "
      f"90-day: {labels[:,2].mean()*100:.1f}%)")

# Flatten across patients and time steps
flat = X_base.reshape(-1, F)

print("\n── Descriptive Statistics ──────────────────────────────────")
print(f"  {'Feature':<14} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  In range?")
for j, name in enumerate(FEATURE_NAMES):
    col  = flat[:, j]
    m, s = col.mean(), col.std()
    mn, mx = col.min(), col.max()
    lo, hi = REF_RANGES[name]
    ok = "✓" if mn >= lo - 0.5 and mx <= hi + 0.5 else "✗"
    print(f"  {name:<14} {m:>8.3f} {s:>8.3f} {mn:>8.3f} {mx:>8.3f}  {ok}")

print("\n── Correlation with 7-day Risk Label ────────────────────────")
label_7 = labels[:, 0]                   # (N,)
mean_per_patient = X_base.mean(axis=1)   # (N, F)

print(f"  {'Feature':<14} {'Pearson r':>10}  {'p-value':>12}")
for j, name in enumerate(FEATURE_NAMES):
    r, p = pearsonr(mean_per_patient[:, j], label_7)
    print(f"  {name:<14} {r:>10.4f}  {p:>12.4e}")

print("\n── Normality Check (Shapiro-Wilk, sample n=200) ────────────")
rng     = np.random.default_rng(42)
sample  = flat[rng.choice(len(flat), size=min(200, len(flat)), replace=False)]
print(f"  {'Feature':<14} {'W':>8}  {'p-value':>12}  {'Normal?':>10}")
for j, name in enumerate(FEATURE_NAMES):
    w, p = shapiro(sample[:, j])
    print(f"  {name:<14} {w:>8.4f}  {p:>12.4e}  "
          + ("Yes" if p > 0.05 else "No (skewed)"))

print("\n" + "="*60)
print(" Validation complete.")
print("="*60 + "\n")
