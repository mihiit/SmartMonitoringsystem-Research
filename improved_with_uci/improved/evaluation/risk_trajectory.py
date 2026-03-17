"""
risk_trajectory.py
==================
Risk trajectory plots — baseline vs intervention, with uncertainty bands.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_risk_trajectory(
    model,
    X_base,
    X_interv,
    users,
    mean_preds=None,
    std_preds=None,
    patient: int = 0,
    output_idx: int = 0,
    save_path: str = "figures/risk_trajectory.png",
):
    """
    Plot baseline vs intervention risk for a single patient,
    optionally with uncertainty bands.

    Parameters
    ----------
    model      : trained model
    X_base     : (N, T, F) baseline tensor
    X_interv   : (N, T, F) intervention tensor
    users      : (N,) user IDs
    mean_preds : (N, output_dim) MC-Dropout means (optional)
    std_preds  : (N, output_dim) MC-Dropout stds  (optional)
    patient    : index of patient to plot
    output_idx : 0=7-day, 1=30-day, 2=90-day
    """
    import torch

    model.eval()
    with torch.no_grad():
        R_base   = model(X_base,   users).numpy()
        R_interv = model(X_interv, users).numpy()

    # 3-point risk values for this patient
    horizons = [7, 30, 90]
    rb = R_base[patient]
    ri = R_interv[patient]

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(horizons, rb, "o-",  color="#2196F3", linewidth=2,
            markersize=7, label="Baseline trajectory")
    ax.plot(horizons, ri, "s--", color="#F44336", linewidth=2,
            markersize=7, label="Post-intervention trajectory")

    # Uncertainty bands (if available)
    if mean_preds is not None and std_preds is not None:
        sig = std_preds[patient]
        ax.fill_between(horizons,
                        rb - 1.96 * sig, rb + 1.96 * sig,
                        alpha=0.15, color="#2196F3", label="95% CI (baseline)")
        ax.fill_between(horizons,
                        ri - 1.96 * sig, ri + 1.96 * sig,
                        alpha=0.15, color="#F44336", label="95% CI (intervention)")

    ax.set_xlabel("Days Ahead", fontsize=12)
    ax.set_ylabel("Predicted Risk Probability", fontsize=12)
    ax.set_title(f"Risk Trajectory — Patient {patient}", fontsize=13)
    ax.set_ylim(0, 1)
    ax.set_xticks(horizons)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")

    return {"baseline": rb.tolist(), "intervention": ri.tolist(),
            "delta": (ri - rb).tolist()}
