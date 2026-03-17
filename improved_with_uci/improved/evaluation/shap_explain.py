"""
shap_explain.py
===============
Temporal SHAP explainability — baseline and post-intervention attribution.

Improvements over original:
  - Computes SHAP for BOTH baseline and intervention inputs
  - Reports Δφ (attribution shift) per feature (eq. 38-39 in paper)
  - Separates temporal aggregation from output aggregation
  - Saves two figures: overall importance + intervention delta
"""

import numpy as np
import matplotlib.pyplot as plt

FEATURE_NAMES = ["BP", "Glucose", "BMI", "Activity", "AgeFactor", "CV_Risk"]


def explain(
    model,
    X_base_np:   np.ndarray,
    X_interv_np: np.ndarray,
    users_np:    np.ndarray,
    n_background: int = 50,
    n_explain:    int = 20,
    save_dir:     str = "figures",
):
    """
    Parameters
    ----------
    X_base_np    : (N, T, F) numpy  — baseline trajectories
    X_interv_np  : (N, T, F) numpy  — intervention trajectories
    users_np     : (N,) numpy int
    n_background : SHAP background samples
    n_explain    : samples to explain
    save_dir     : where to save figures
    """
    import torch, shap

    T = X_base_np.shape[1]
    F = X_base_np.shape[2]

    # Flatten time × feature for SHAP (it works on 2D)
    X_base_flat   = X_base_np[:n_explain + n_background].reshape(-1, T * F)
    X_interv_flat = X_interv_np[:n_explain + n_background].reshape(-1, T * F)

    background = X_base_flat[:n_background]
    explain_b  = X_base_flat[n_background: n_background + n_explain]
    explain_i  = X_interv_flat[n_background: n_background + n_explain]

    fixed_users = torch.tensor(users_np[:1]).repeat(n_explain)

    def _predict(x_flat):
        x_3d = torch.tensor(x_flat.reshape(-1, T, F)).float()
        u    = fixed_users[:len(x_flat)]
        with torch.no_grad():
            return model(x_3d, u).numpy()

    explainer = shap.Explainer(_predict, background)

    sv_base   = explainer(explain_b).values    # (n, T*F, output_dim)
    sv_interv = explainer(explain_i).values

    # Reshape to (n, T, F, output_dim) then average over time and outputs
    sv_b = np.abs(sv_base).reshape(n_explain, T, F, -1).mean(axis=(0, 2, 3))   # (T, F) → (F,)
    sv_b = np.abs(sv_base).reshape(n_explain, T, F, -1).mean(axis=(0, 1, 3))   # (F,)
    sv_i = np.abs(sv_interv).reshape(n_explain, T, F, -1).mean(axis=(0, 1, 3))

    delta = sv_i - sv_b   # Δφ per feature

    # ── Figure 1: Overall feature importance (baseline) ───────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2196F3" if v >= 0 else "#F44336" for v in sv_b]
    ax.bar(FEATURE_NAMES, sv_b, color=colors)
    ax.set_title("SHAP Feature Importance (Baseline)", fontsize=13)
    ax.set_ylabel("Mean |SHAP|", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/shap_importance.png", dpi=150)
    plt.close(fig)

    # ── Figure 2: Attribution shift under intervention ────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    bar_colors = ["#4CAF50" if d > 0 else "#FF5722" for d in delta]
    ax.bar(FEATURE_NAMES, delta, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("SHAP Attribution Shift Under Intervention (Δφ)", fontsize=13)
    ax.set_ylabel("Δ Mean |SHAP|", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{save_dir}/shap_intervention_delta.png", dpi=150)
    plt.close(fig)

    print(f"\n─── SHAP Attribution ──────────────────────────────────────")
    print(f"  {'Feature':<14} {'Baseline':>10}  {'Interv.':>10}  {'Δφ':>10}")
    for fn, b, iv, d in zip(FEATURE_NAMES, sv_b, sv_i, delta):
        print(f"  {fn:<14} {b:>10.4f}  {iv:>10.4f}  {d:>+10.4f}")
    print(f"────────────────────────────────────────────────────────────\n")

    return {
        "baseline_importance":     dict(zip(FEATURE_NAMES, sv_b.tolist())),
        "intervention_importance": dict(zip(FEATURE_NAMES, sv_i.tolist())),
        "delta_phi":               dict(zip(FEATURE_NAMES, delta.tolist())),
    }
