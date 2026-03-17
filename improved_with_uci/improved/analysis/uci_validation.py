"""
uci_validation.py
=================
Full validation on UCI real data — generates all metrics and figures
for the real-data section of the paper.

Run AFTER train_uci.py has saved the model .pth files.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.uci_diabetes_loader import load_uci_diabetes, load_uci_full
from utils.uci_temporal import generate_uci_temporal
from utils.metrics import (auc_with_ci, expected_calibration_error,
                            behavioural_robustness, intervention_significance,
                            uncertainty_summary)
from models.transformer_model import TransformerModel
from evaluation.uncertainty import predict_with_uncertainty

os.makedirs("figures", exist_ok=True)
UCI_INPUT_DIM = 6

def to_t(a):  return torch.tensor(a).float()
def to_ti(a): return torch.tensor(a).long()

print("\n" + "="*60)
print("  UCI Diabetes Real-Data Validation")
print("="*60)

# ── Load data ──────────────────────────────────────────────────────────────
X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, ids = load_uci_diabetes()
_, _, yte3, ute = generate_uci_temporal(X_te, y_te)
Xtr_b, _, ytr3, utr = generate_uci_temporal(X_tr, y_tr)

Xte_b, Xte_i, yte3, ute = generate_uci_temporal(X_te, y_te)
Xte_t  = to_t(Xte_b); Xte_it = to_t(Xte_i)
yte_t  = to_t(yte3);  ute_t  = to_ti(ute)

num_users = len(utr)

# ── Load model ──────────────────────────────────────────────────────────────
trans = TransformerModel(input_dim=UCI_INPUT_DIM, num_users=num_users,
                          use_personalization=True, dropout=0.20)
if os.path.exists("uci_transformer.pth"):
    trans.load_state_dict(torch.load("uci_transformer.pth", map_location="cpu"))
    trans.eval()
    print("  Loaded uci_transformer.pth")
else:
    print("  WARNING: uci_transformer.pth not found — run train_uci.py first")
    sys.exit(1)

# ── Predictive performance ──────────────────────────────────────────────────
with torch.no_grad():
    R_b = trans(Xte_t,  ute_t).numpy()
    R_i = trans(Xte_it, ute_t).numpy()

y_true = yte_t[:, 0].numpy()
auc, lo, hi = auc_with_ci(y_true, R_b[:, 0], n_boot=500)
ece = expected_calibration_error(y_true, R_b[:, 0])

print(f"\n  AUC  (7-day, test): {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"  ECE:                {ece:.4f}")

# ── Intervention sensitivity ────────────────────────────────────────────────
br  = behavioural_robustness(R_b[:, 0], R_i[:, 0])
sig = intervention_significance(R_b[:, 0], R_i[:, 0])

print(f"\n  Mean risk (baseline):     {R_b[:,0].mean():.4f}")
print(f"  Mean risk (intervention): {R_i[:,0].mean():.4f}")
print(f"  Absolute ΔR:              {sig['mean_delta']:+.6f}")
print(f"  Relative change:          {sig['relative_change_%']:+.3f}%")
print(f"  BR:                       {br['BR']:.6f}")
print(f"  BR_velocity:              {br['BR_velocity']:.6f}")
print(f"  Wilcoxon p-value:         {sig['p_value']:.4e}")

# ── MC-Dropout uncertainty ──────────────────────────────────────────────────
mean_u, std_u, unc = predict_with_uncertainty(trans, Xte_t, ute_t, n_runs=50)
print(f"\n  Uncertainty σ (mean):     {unc['mean_std']:.4f}")
print(f"  CoV:                      {unc['CoV_%']:.2f}%")

# ── Figure: Glucose trajectory for a real patient ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Plot 1: Real glucose trajectory for patient 0
raw_glucose = Xte_b[0, :, 0]   # normalised glucose feature
t = np.arange(len(raw_glucose))
axes[0].plot(t, raw_glucose, color='#2196F3', linewidth=2, label='Real glucose (norm.)')
axes[0].axvline(x=10, color='grey', linestyle=':', linewidth=1.5, label='Intervention start')
axes[0].set_xlabel('Day', fontsize=11)
axes[0].set_ylabel('Normalised glucose', fontsize=11)
axes[0].set_title('UCI Patient: Real Glucose Trajectory', fontsize=12)
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

# Plot 2: Baseline vs intervention risk at 7/30/90 days
horizons = [7, 30, 90]
x = np.arange(len(horizons)); w = 0.35
axes[1].bar(x - w/2, R_b.mean(axis=0), w, label='Baseline',
            color='#2196F3', alpha=0.85)
axes[1].bar(x + w/2, R_i.mean(axis=0), w, label='Intervention',
            color='#4CAF50', alpha=0.85)
axes[1].set_xticks(x); axes[1].set_xticklabels(['7-day', '30-day', '90-day'])
axes[1].set_ylabel('Mean Predicted Risk', fontsize=11)
axes[1].set_title('UCI: Intervention Effect on Risk', fontsize=12)
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=9); axes[1].grid(axis='y', alpha=0.3)
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('figures/uci_validation.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n  Saved: figures/uci_validation.png")

# ── Summary table for paper ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  UCI Real-Data Summary (for paper Table)")
print("="*60)
print(f"  Dataset          : UCI Diabetes (real IDDM patients, N=70)")
print(f"  Test set         : N=11 patients")
print(f"  Features         : 6 (glucose, CV, insulin, activity, meal, hypo)")
print(f"  AUC              : {auc:.4f}  [{lo:.4f}, {hi:.4f}]")
print(f"  ECE              : {ece:.4f}")
print(f"  BR               : {br['BR']:.6f}")
print(f"  BR_velocity      : {br['BR_velocity']:.6f}")
print(f"  Wilcoxon p       : {sig['p_value']:.4e}")
print(f"  Mean ΔR          : {sig['mean_delta']:+.6f}  ({sig['relative_change_%']:+.3f}%)")
print(f"  Uncertainty σ    : {unc['mean_std']:.4f}")
print(f"  CoV              : {unc['CoV_%']:.2f}%")
print("="*60 + "\n")
