"""
cross_dataset_validation.py
============================
Compares model behaviour across two datasets:
  1. Pima Indians Diabetes (N=768, 6 features, single-sex cohort)
  2. NHANES-calibrated synthetic (N=2000, 8 features, mixed population)

This addresses the top-tier journal requirement for generalizability
validation beyond a single dataset.

Reports:
  - AUC comparison across datasets
  - Intervention sensitivity (BR) comparison
  - Uncertainty profile comparison
  - Discussion of consistency and divergence
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils.preprocessing import load_pima
from utils.clinical_temporal import generate_clinical_temporal
from utils.nhanes_synthetic import load_nhanes_synthetic
from utils.nhanes_temporal import generate_nhanes_temporal
from utils.metrics import (auc_with_ci, expected_calibration_error,
                            behavioural_robustness, intervention_significance,
                            uncertainty_summary)
from models.transformer_model import TransformerModel
from evaluation.uncertainty import predict_with_uncertainty

plt.rcParams.update({"font.family": "sans-serif", "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

os.makedirs("figures", exist_ok=True)

def to_t(a):  return torch.tensor(a).float()
def to_ti(a): return torch.tensor(a).long()


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PIMA RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Cross-Dataset Validation")
print("="*60)

print("\n[1/2] Pima Indians dataset...")
X_tr_p, _, X_te_p, y_tr_p, _, y_te_p, _ = load_pima()

# Single call → returns (X_base, X_interv, y, users)
X_te_seq_p, X_te_int_p, y_te_lbl_p, u_te_p = generate_clinical_temporal(
    X_te_p, y_te_p, seq_len=30)
X_te_pt     = to_t(X_te_seq_p);  X_te_int_pt = to_t(X_te_int_p)
y_te_pt     = to_t(y_te_lbl_p);  u_te_pt     = to_ti(u_te_p)

tr_b_p, tr_i_p, y_tr_lbl_p, u_tr_p = generate_clinical_temporal(X_tr_p, y_tr_p)
num_users_p = len(u_tr_p)

trans_p = TransformerModel(input_dim=6, num_users=num_users_p,
                            use_personalization=True, dropout=0.20)
if os.path.exists("transformer_model.pth"):
    trans_p.load_state_dict(torch.load("transformer_model.pth", map_location="cpu"))
    trans_p.eval()
    pima_loaded = True
    print("  Loaded transformer_model.pth")
else:
    pima_loaded = False
    print("  WARNING: transformer_model.pth not found — skipping Pima eval")


# ═══════════════════════════════════════════════════════════════════════════
# 2.  NHANES RESULTS
# ═══════════════════════════════════════════════════════════════════════════

print("\n[2/2] NHANES-calibrated dataset...")
X_tr_n, _, X_te_n, y_tr_n, _, y_te_n, _ = load_nhanes_synthetic()
X_te_b_n, X_te_i_n, y_te_lbl_n, u_te_n = generate_nhanes_temporal(
    X_te_n, y_te_n, seq_len=30)
X_tr_b_n, X_tr_i_n, y_tr_lbl_n, u_tr_n = generate_nhanes_temporal(
    X_tr_n, y_tr_n, seq_len=30)
num_users_n = len(u_tr_n)

Xte_nt   = to_t(X_te_b_n); Xte_int_nt = to_t(X_te_i_n)
yte_nt   = to_t(y_te_lbl_n); ute_nt   = to_ti(u_te_n)

trans_n = TransformerModel(input_dim=8, num_users=num_users_n,
                            use_personalization=True, dropout=0.20)
if os.path.exists("nhanes_transformer.pth"):
    trans_n.load_state_dict(torch.load("nhanes_transformer.pth", map_location="cpu"))
    trans_n.eval()
    nhanes_loaded = True
    print("  Loaded nhanes_transformer.pth")
else:
    nhanes_loaded = False
    print("  WARNING: nhanes_transformer.pth not found — run training/train_nhanes.py first")


# ═══════════════════════════════════════════════════════════════════════════
# 3.  COMPUTE METRICS FOR EACH DATASET
# ═══════════════════════════════════════════════════════════════════════════

dataset_results = {}

for tag, model, Xte_b, Xte_i, yte_t, ute_t, loaded in [
    ("Pima (6-feature)",         trans_p, X_te_pt,  X_te_int_pt, y_te_pt,  u_te_pt,  pima_loaded),
    ("NHANES-calibrated (8-ft)", trans_n, Xte_nt,   Xte_int_nt,  yte_nt,   ute_nt,   nhanes_loaded),
]:
    if not loaded:
        print(f"\n  Skipping {tag} (model not loaded)")
        continue

    print(f"\n  Computing metrics: {tag}")

    # AUC
    with torch.no_grad():
        R_b = model(Xte_b, ute_t).numpy()
        R_i = model(Xte_i, ute_t).numpy()

    y_true  = yte_t[:, 0].numpy()
    auc, lo, hi = auc_with_ci(y_true, R_b[:, 0])
    ece = expected_calibration_error(y_true, R_b[:, 0])

    # Intervention sensitivity
    br  = behavioural_robustness(R_b[:, 0], R_i[:, 0])
    sig = intervention_significance(R_b[:, 0], R_i[:, 0])

    # Uncertainty
    mean_u, std_u, unc = predict_with_uncertainty(model, Xte_b, ute_t, n_runs=30)

    dataset_results[tag] = {
        "AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece,
        "BR":  br["BR"], "BR_velocity": br["BR_velocity"],
        "mean_delta": sig["mean_delta"],
        "p_value":    sig["p_value"],
        "relative_%": sig["relative_change_%"],
        "uncertainty_sigma": unc["mean_std"],
        "CoV_%": unc["CoV_%"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4.  PRINT COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("  Cross-Dataset Comparison: Transformer (Personalized)")
print("="*70)
print(f"\n  {'Metric':<35}", end="")
for tag in dataset_results:
    print(f"  {tag:<30}", end="")
print()
print("  " + "-"*80)

metric_labels = [
    ("AUC (7-day risk)",        "AUC"),
    ("95% CI lower",            "CI_lo"),
    ("95% CI upper",            "CI_hi"),
    ("ECE (calibration error)", "ECE"),
    ("BR (behavioural robust.)", "BR"),
    ("BR_velocity",             "BR_velocity"),
    ("Mean ΔR (intervention)",  "mean_delta"),
    ("Intervention p-value",    "p_value"),
    ("Relative change (%)",     "relative_%"),
    ("Uncertainty σ (mean)",    "uncertainty_sigma"),
    ("CoV (%)",                 "CoV_%"),
]

for label, key in metric_labels:
    print(f"  {label:<35}", end="")
    for tag, r in dataset_results.items():
        val = r[key]
        if key == "p_value":
            print(f"  {val:<30.4e}", end="")
        elif key in ("relative_%", "CoV_%"):
            print(f"  {val:<30.3f}", end="")
        elif key == "mean_delta":
            print(f"  {val:<+30.6f}", end="")
        else:
            print(f"  {val:<30.4f}", end="")
    print()

print("\n" + "="*70)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  FIGURE: Side-by-side AUC comparison
# ═══════════════════════════════════════════════════════════════════════════

if len(dataset_results) == 2:
    tags  = list(dataset_results.keys())
    aucs  = [dataset_results[t]["AUC"]   for t in tags]
    ci_lo = [dataset_results[t]["CI_lo"] for t in tags]
    ci_hi = [dataset_results[t]["CI_hi"] for t in tags]
    brs   = [dataset_results[t]["BR"]    for t in tags]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # AUC
    yerr = [[a-lo for a, lo in zip(aucs, ci_lo)],
            [hi-a for a, hi in zip(aucs, ci_hi)]]
    axes[0].bar(tags, aucs, color=["#2196F3", "#4CAF50"], width=0.5, zorder=3)
    axes[0].errorbar(range(2), aucs, yerr=yerr,
                     fmt="none", color="black", capsize=6, zorder=4)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("AUC (7-day risk)")
    axes[0].set_title("AUC Across Datasets"); axes[0].grid(axis="y", alpha=0.3, zorder=0)
    for i, (a, tag) in enumerate(zip(aucs, tags)):
        axes[0].text(i, a + 0.03, f"{a:.3f}", ha="center", fontsize=10)

    # BR
    axes[1].bar(tags, brs, color=["#FF9800", "#9C27B0"], width=0.5, zorder=3)
    axes[1].set_ylabel("Behavioural Robustness (BR)")
    axes[1].set_title("Intervention Sensitivity Across Datasets")
    axes[1].grid(axis="y", alpha=0.3, zorder=0)
    for i, (b, tag) in enumerate(zip(brs, tags)):
        axes[1].text(i, b + 0.0001, f"{b:.5f}", ha="center", fontsize=9)

    plt.suptitle("Cross-Dataset Generalisation Validation", fontsize=13,
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig("figures/cross_dataset_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("\n  Saved: figures/cross_dataset_comparison.png")

print("\n  Cross-dataset validation complete.\n")
