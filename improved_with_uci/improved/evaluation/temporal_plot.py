"""
temporal_plot.py
================
Plot temporal feature evolution for a representative patient.
"""

import numpy as np
import matplotlib.pyplot as plt


FEATURE_NAMES = ["BP", "Glucose", "BMI", "Activity", "AgeFactor", "CV Risk"]
COLORS = ["#2196F3", "#F44336", "#FF9800", "#4CAF50", "#9C27B0", "#795548"]


def plot_temporal_features(
    X: np.ndarray,
    patient: int = 0,
    save_path: str = "figures/temporal_evolution.png",
):
    """
    X : (N, T, F) numpy array
    """
    sample = X[patient]           # (T, F)
    T = sample.shape[0]
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (name, color) in enumerate(zip(FEATURE_NAMES, COLORS)):
        ax.plot(t, sample[:, j], label=name, color=color, linewidth=1.8)

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Normalised Value", fontsize=12)
    ax.set_title(f"Temporal Feature Evolution — Patient {patient}", fontsize=13)
    ax.legend(fontsize=9, ncol=3)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_baseline_vs_intervention_features(
    X_base:   np.ndarray,
    X_interv: np.ndarray,
    patient:  int = 0,
    feature:  int = 3,          # default: Activity
    save_path: str = "figures/intervention_feature_evolution.png",
):
    """
    Compare one feature trajectory across baseline and intervention branches.
    """
    b = X_base[patient, :, feature]
    iv = X_interv[patient, :, feature]
    T = len(b)
    t = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, b,  "o-", color="#2196F3", linewidth=2, markersize=3,
            label=f"{FEATURE_NAMES[feature]} — Baseline")
    ax.plot(t, iv, "s--", color="#F44336", linewidth=2, markersize=3,
            label=f"{FEATURE_NAMES[feature]} — Intervention")
    ax.axvline(x=10, color="grey", linestyle=":", linewidth=1.5,
               label="Intervention start (t=10)")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Feature Value (normalised)", fontsize=12)
    ax.set_title(f"Feature Evolution: {FEATURE_NAMES[feature]}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
