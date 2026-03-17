"""
risk_velocity.py
================
Risk velocity across the full temporal horizon.

FIX over original:
  - Original had only 3 data points (7/30/90 day output), giving just
    2 velocity values — meaningless for trajectory analysis.
  - New version operates on the full (N, T) risk sequence obtained by
    running the model on each time step's cumulative window.
  - classify_velocity now returns a structured dict for cleaner reporting.
"""

import numpy as np


def compute_velocity(risk_sequence):
    """
    Compute per-step velocity for a 1-D sequence of length T.
    Returns dict with full velocity array + summary stats.
    """
    r = np.asarray(risk_sequence, dtype=float)
    v = np.diff(r)                          # length T-1

    return {
        "velocity":           v,
        "mean_velocity":      float(v.mean()),
        "max_velocity":       float(v.max()),
        "min_velocity":       float(v.min()),
        "velocity_std":       float(v.std()),
        "acceleration":       float(np.diff(v).mean()) if len(v) > 1 else 0.0,
    }


def classify_velocity(v_mean: float) -> dict:
    """Categorise progression speed from mean velocity."""
    if v_mean < -0.005:
        label = "Improving"
        severity = 0
    elif -0.005 <= v_mean < 0.001:
        label = "Stable"
        severity = 1
    elif 0.001 <= v_mean < 0.010:
        label = "Slowly Worsening"
        severity = 2
    elif 0.010 <= v_mean < 0.030:
        label = "Worsening"
        severity = 3
    else:
        label = "Rapid Risk Increase"
        severity = 4

    return {"label": label, "severity": severity, "mean_velocity": v_mean}


def build_temporal_risk_series(model, X, users, output_idx: int = 0):
    """
    Build a (N, T) risk matrix by feeding cumulative sub-sequences.
    This gives a true temporal risk trajectory rather than the 3-point output.

    NOTE: computationally heavier — use for a representative subset.
    """
    import torch

    model.eval()
    N, T, F = X.shape
    risk_matrix = np.zeros((N, T), dtype=np.float32)

    with torch.no_grad():
        for t in range(1, T + 1):
            X_t = X[:, :t, :]          # (N, t, F)
            # Pad to full seq_len so positional encoding stays consistent
            pad = torch.zeros(N, T - t, F)
            X_pad = torch.cat([X_t, pad], dim=1)
            pred = model(X_pad, users)
            risk_matrix[:, t - 1] = pred[:, output_idx].numpy()

    return risk_matrix   # (N, T)
