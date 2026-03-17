"""
cv_plot.py
==========
Cardiovascular risk progression plot.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_cv_risk(
    X: np.ndarray,
    patient: int = 0,
    save_path: str = "figures/cv_risk.png",
):
    """
    X : (N, T, F) or (T, F) array — CV risk is feature index 5.
    """
    if X.ndim == 3:
        sample = X[patient]
    else:
        sample = X

    cv = sample[:, 5]
    T  = len(cv)
    t  = np.arange(T)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, cv, color="#E91E63", linewidth=2)
    ax.fill_between(t, cv.min(), cv, alpha=0.15, color="#E91E63")

    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("CV Risk (normalised)", fontsize=12)
    ax.set_title(f"Cardiovascular Risk Progression — Patient {patient}", fontsize=13)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")
