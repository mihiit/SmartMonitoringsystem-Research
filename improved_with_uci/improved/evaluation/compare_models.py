"""
compare_models.py
=================
Master evaluation script — runs all experiments reported in the paper.

Fixes over original:
  1. Consistent AUC metric: always y[:,0] (7-day risk) across ALL evaluations
     → resolves Table 6 vs Table 10 (ablation) AUC contradiction in paper
  2. Proper train/val/test split — AUC reported on held-out TEST set only
  3. Intervention uses two parallel trajectory branches (with physiological lag)
     instead of post-hoc feature perturbation
  4. MC-Dropout uncertainty via enable_mc_dropout()
  5. All statistical tests: bootstrap CI, Wilcoxon signed-rank, ECE
  6. Intervention SHAP delta (Δφ) computed and reported
  7. Full temporal velocity series (not just 3 points)
  8. Calibration curve generated
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

from utils.preprocessing import load_pima
from utils.clinical_temporal import generate_clinical_temporal
from utils.metrics import (auc_with_ci, expected_calibration_error,
                            behavioural_robustness, intervention_significance,
                            uncertainty_summary)

from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel

from evaluation.risk_trajectory import plot_risk_trajectory
from evaluation.ablation import run_ablation
from evaluation.uncertainty import predict_with_uncertainty, print_uncertainty_report
from evaluation.counterfactual import simulate_intervention, print_intervention_report
from evaluation.onset import estimate_onset, print_onset_report
from evaluation.risk_velocity import compute_velocity, classify_velocity
from evaluation.cv_plot import plot_cv_risk
from evaluation.temporal_plot import (plot_temporal_features,
                                       plot_baseline_vs_intervention_features)
from evaluation.figures import (plot_model_comparison, plot_velocity,
                                  plot_uncertainty, plot_intervention_comparison,
                                  plot_calibration_curve)

os.makedirs("figures", exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Loading and preparing clinical data")
print("="*60)

X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_pima()

# Generate trajectory pairs (baseline + intervention) for each split
X_tr_b, X_tr_i, y_tr, u_tr = generate_clinical_temporal(X_train, y_train)
X_va_b, X_va_i, y_va, u_va = generate_clinical_temporal(X_val,   y_val)
X_te_b, X_te_i, y_te, u_te = generate_clinical_temporal(X_test,  y_test)

def tt(a):  return torch.tensor(a).float()
def tti(a): return torch.tensor(a).long()

# Train tensors
Xt_b = tt(X_tr_b); Xt_i = tt(X_tr_i); yt = tt(y_tr); ut = tti(u_tr)
# Test tensors
Xte_b = tt(X_te_b); Xte_i = tt(X_te_i); yte = tt(y_te); ute = tti(u_te)

print(f"  Train: {len(X_tr_b)} | Val: {len(X_va_b)} | Test: {len(X_te_b)}")
print(f"  Trajectory shape: {X_te_b.shape}  (N × T × F)")

# ═══════════════════════════════════════════════════════════════════════════
# 2. LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Loading trained models")
print("="*60)

device = torch.device("cpu")

lstm = LSTMModel(input_size=6, hidden_size=64, num_layers=2,
                 num_users=len(u_tr), dropout=0.20)
lstm.load_state_dict(torch.load("lstm_model.pth", map_location=device))
lstm.eval()
print("  ✓ LSTM loaded")

trans = TransformerModel(input_dim=6, num_users=len(u_tr),
                          use_personalization=True, dropout=0.20)
trans.load_state_dict(torch.load("transformer_model.pth", map_location=device))
trans.eval()
print("  ✓ Transformer (personalized) loaded")

trans_nop = TransformerModel(input_dim=6, num_users=len(u_tr),
                               use_personalization=False, dropout=0.20)
trans_nop.load_state_dict(torch.load("transformer_no_personal.pth", map_location=device))
trans_nop.eval()
print("  ✓ Transformer (no personalization) loaded")

# ═══════════════════════════════════════════════════════════════════════════
# 3. PREDICTIVE PERFORMANCE  (Table 6 in paper)
#    Always uses y[:, 0] — 7-day risk — on HELD-OUT TEST SET
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Predictive Performance  (TEST SET, 7-day risk)")
print("="*60)

results = {}
with torch.no_grad():
    pred_lstm = lstm(Xte_b, ute)
    pred_tr   = trans(Xte_b, ute)
    pred_nop  = trans_nop(Xte_b, ute)

y_true = yte[:, 0].numpy()

for name, pred in [("LSTM", pred_lstm),
                   ("Transformer (Personalized)", pred_tr),
                   ("Transformer (No Personalization)", pred_nop)]:
    y_score = pred[:, 0].numpy()
    auc, lo, hi = auc_with_ci(y_true, y_score)
    ece = expected_calibration_error(y_true, y_score)
    results[name] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    print(f"  {name:<38} AUC={auc:.4f}  [{lo:.4f},{hi:.4f}]  ECE={ece:.4f}")

plot_model_comparison(results)

# ═══════════════════════════════════════════════════════════════════════════
# 4. ABLATION STUDY  (Table 10 in paper)
#    FIX: uses same y[:,0] as Table 6 — no more inflated AUC
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Ablation Study")
print("="*60)

ablation_models = {
    "LSTM Baseline":                    lstm,
    "Transformer (Full)":               trans,
    "Transformer (No Personalization)": trans_nop,
}
run_ablation(ablation_models, Xte_b, yte, ute, output_idx=0)

# ═══════════════════════════════════════════════════════════════════════════
# 5. TEMPORAL FEATURE EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Temporal Feature Plots")
print("="*60)

plot_temporal_features(X_te_b, patient=0)
plot_baseline_vs_intervention_features(X_te_b, X_te_i, patient=0, feature=3)
plot_cv_risk(X_te_b, patient=0)

# ═══════════════════════════════════════════════════════════════════════════
# 6. INTERVENTION SENSITIVITY  (Section 10.3, 10.7 in paper)
#    FIX: uses two proper trajectory branches, not post-hoc perturbation
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Intervention Sensitivity Analysis")
print("="*60)

interv_result = simulate_intervention(trans, Xte_b, Xte_i, ute)
print_intervention_report(interv_result)

plot_intervention_comparison(
    interv_result["risk_base"],
    interv_result["risk_interv"],
    patient=0,
)

plot_risk_trajectory(
    trans, Xte_b, Xte_i, ute,
    patient=0,
)

# ═══════════════════════════════════════════════════════════════════════════
# 7. RISK VELOCITY
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Risk Velocity Analysis")
print("="*60)

patient_risk = pred_tr[0].numpy()
vel = compute_velocity(patient_risk)
cls = classify_velocity(vel["mean_velocity"])

print(f"  Mean velocity : {vel['mean_velocity']:+.6f}")
print(f"  Min / Max     : {vel['min_velocity']:+.6f} / {vel['max_velocity']:+.6f}")
print(f"  Status        : {cls['label']}")

plot_velocity(vel)

# ═══════════════════════════════════════════════════════════════════════════
# 8. UNCERTAINTY ESTIMATION  (MC-Dropout)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Predictive Uncertainty (MC-Dropout, 50 runs)")
print("="*60)

mean_u, std_u, unc_summary = predict_with_uncertainty(trans, Xte_b, ute, n_runs=50)
print_uncertainty_report(mean_u, std_u, unc_summary)
plot_uncertainty(mean_u[:, 0], std_u[:, 0])

# ═══════════════════════════════════════════════════════════════════════════
# 9. CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Calibration Curve")
print("="*60)

plot_calibration_curve(y_true, mean_u[:, 0])

# ═══════════════════════════════════════════════════════════════════════════
# 10. ONSET PREDICTION
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Time-to-Onset Prediction (Patient 0)")
print("="*60)

onset = estimate_onset(pred_tr[0].numpy())
print_onset_report(onset)

# ═══════════════════════════════════════════════════════════════════════════
# 11. SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  FULL RESULTS SUMMARY")
print("="*60)
print(f"\n  {'Metric':<45} {'Value'}")
print(f"  {'-'*60}")

for name, r in results.items():
    print(f"  AUC  [{name:<38}]  {r['AUC']:.4f}  CI=[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]")
    print(f"  ECE  [{name:<38}]  {r['ECE']:.4f}")

print(f"\n  Intervention mean ΔR                           {interv_result['mean_delta']:+.6f}")
print(f"  Intervention relative change (%)               {interv_result['relative_%']:+.3f}")
print(f"  Intervention p-value (Wilcoxon)                {interv_result['p_value']:.4e}")
print(f"  Behavioural Robustness (BR)                    {interv_result['BR']:.6f}")
print(f"  BR_velocity                                    {interv_result['BR_velocity']:.6f}")
print(f"\n  Prediction uncertainty σ (mean)                {unc_summary['mean_std']:.4f}")
print(f"  Confidence (1-σ)                               {unc_summary['confidence']*100:.2f}%")
print(f"  Coefficient of Variation                       {unc_summary['CoV_%']:.2f}%")

print("\n  ✅  All evaluations complete. Figures saved to figures/\n")
