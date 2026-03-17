"""
figures.py
==========
Publication-quality figure generation for all paper results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

BLUE   = "#2196F3"
RED    = "#F44336"
GREEN  = "#4CAF50"
GREY   = "#9E9E9E"
ORANGE = "#FF9800"


def plot_model_comparison(results: dict, save_path="figures/model_comparison.png"):
    """
    Bar chart with 95% CI whiskers.
    results: {name: {"AUC": float, "CI_lo": float, "CI_hi": float}}
    """
    names  = list(results.keys())
    aucs   = [results[n]["AUC"]   for n in names]
    ci_lo  = [results[n]["CI_lo"] for n in names]
    ci_hi  = [results[n]["CI_hi"] for n in names]
    yerr   = [[a - lo for a, lo in zip(aucs, ci_lo)],
              [hi - a for a, hi in zip(aucs, ci_hi)]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, aucs, color=[BLUE, GREEN, ORANGE],
                  width=0.5, zorder=3)
    ax.errorbar(range(len(names)), aucs, yerr=yerr,
                fmt="none", color="black", capsize=5, linewidth=1.5, zorder=4)

    ax.set_ylim(0, 1)
    ax.set_ylabel("AUC (7-day risk)", fontsize=12)
    ax.set_title("Model Performance Comparison (AUC)", fontsize=13)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, auc + 0.02,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_velocity(velocity_dict: dict, save_path="figures/risk_velocity.png"):
    """
    Plot full velocity series over time.
    velocity_dict: output of compute_velocity()
    """
    v = velocity_dict["velocity"]
    t = np.arange(1, len(v) + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, v, color=BLUE, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.fill_between(t, v, 0,
                    where=(v > 0), alpha=0.25, color=RED,   label="Worsening")
    ax.fill_between(t, v, 0,
                    where=(v < 0), alpha=0.25, color=GREEN, label="Improving")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Risk Velocity (ΔR/Δt)", fontsize=12)
    ax.set_title("Risk Velocity Over Time", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_uncertainty(mean: np.ndarray, std: np.ndarray,
                     save_path="figures/uncertainty.png"):
    """
    Uncertainty across patients — sorted by mean risk.
    mean, std : (N,) arrays for a single output dimension
    """
    order = np.argsort(mean)
    m = mean[order]
    s = std[order]
    x = np.arange(len(m))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, m, color=BLUE, linewidth=1.5, label="Mean predicted risk")
    ax.fill_between(x, m - 1.96*s, m + 1.96*s,
                    alpha=0.20, color=BLUE, label="95% CI (MC-Dropout)")

    ax.set_xlabel("Patient (sorted by risk)", fontsize=12)
    ax.set_ylabel("Predicted Risk", fontsize=12)
    ax.set_title("Predictive Uncertainty Across Patients", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_intervention_comparison(
    R_base:   np.ndarray,
    R_interv: np.ndarray,
    patient:  int = 0,
    save_path: str = "figures/intervention_comparison.png",
):
    """
    Side-by-side 7/30/90-day risk for one patient, baseline vs intervention.
    """
    horizons = [7, 30, 90]
    rb = R_base[patient]
    ri = R_interv[patient]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(horizons))
    w = 0.35

    bars_b = ax.bar(x - w/2, rb, w, label="Baseline",     color=BLUE,  alpha=0.85)
    bars_i = ax.bar(x + w/2, ri, w, label="Intervention", color=GREEN, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}-day" for h in horizons])
    ax.set_ylabel("Predicted Risk", fontsize=12)
    ax.set_title(f"Intervention Impact — Patient {patient}", fontsize=13)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: str = "figures/calibration.png",
):
    """Reliability diagram (calibration curve)."""
    bins   = np.linspace(0, 1, n_bins + 1)
    bin_mid, acc, conf, counts = [], [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        bin_mid.append((lo + hi) / 2)
        acc.append(y_true[mask].mean())
        conf.append(y_prob[mask].mean())
        counts.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot(conf, acc, "o-", color=BLUE, linewidth=2, markersize=7,
            label="Model")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Fraction of Positives",      fontsize=12)
    ax.set_title("Calibration Curve (Reliability Diagram)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
