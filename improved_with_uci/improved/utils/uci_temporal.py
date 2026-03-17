"""
uci_temporal.py
===============
Converts UCI diabetes sequences into the dual-branch trajectory format
(baseline + intervention) matching the rest of the framework.

The UCI data is REAL longitudinal patient data — no simulation needed
for the baseline branch. The intervention branch models what would happen
if exercise activity was increased from day 10 onward.

Features (6):
  0  norm_glucose     Normalised daily mean glucose
  1  glucose_cv       Daily glucose coefficient of variation
  2  insulin_dose     Normalised daily insulin
  3  activity_index   Exercise activity (0=none, 0.5=typical, 1.0=more)
  4  meal_index       Meal behaviour
  5  hypoglycemia_flag Fraction of readings below 70 mg/dL
"""

import numpy as np


def generate_uci_temporal(
    X: np.ndarray,
    y: np.ndarray,
    intervention_start: int   = 10,
    intervention_delta: float = 0.30,
    lag_tau:            float = 5.0,
    noise_std:          float = 0.008,
    random_seed:        int   = 42,
):
    """
    Parameters
    ----------
    X  : (N, T, 6) real UCI sequences from load_uci_diabetes()
    y  : (N,)      binary outcome labels

    Returns
    -------
    X_base   : (N, T, 6) — real trajectories (baseline)
    X_interv : (N, T, 6) — intervention branch (activity raised from t=10)
    y_3      : (N, 3)    — 3-output labels (7/30/90-day proxies from real data)
    users    : (N,)      — patient IDs
    """
    rng = np.random.default_rng(random_seed)
    N, T, F = X.shape

    X_base   = X.copy().astype(np.float32)
    X_interv = X.copy().astype(np.float32)

    # Apply intervention to activity feature (index 3) from t=intervention_start
    for t in range(T):
        if t >= intervention_start:
            ramp = 1.0 - np.exp(-(t - intervention_start) / lag_tau)
            delta = intervention_delta * ramp
            # Intervention raises activity, which attenuates glucose drift
            X_interv[:, t, 3] = np.clip(X_interv[:, t, 3] + delta, 0.0, 1.0)
            # Glucose responds with slight reduction (scaled by delta, lagged)
            glucose_effect = -0.015 * delta
            X_interv[:, t, 0] = X_interv[:, t, 0] + glucose_effect
            # Add small biological noise to intervention branch
            X_interv[:, t, 0] += rng.normal(0, noise_std, N).astype(np.float32)

    # Build 3-output labels:
    # 7-day  → outcome label as-is
    # 30-day → outcome label with small stochastic noise (slightly harder)
    # 90-day → weighted by last-day glucose in sequence
    alpha = 0.70
    last_glucose = X[:, -1, 0]  # final normalised glucose value

    rng2 = np.random.default_rng(random_seed + 1)
    y_7  = y.astype(np.float32)
    y_30 = np.clip(
        alpha * y + (1 - alpha) * (last_glucose > 0).astype(float)
        + rng2.normal(0, 0.05, N), 0, 1
    ).round().astype(np.float32)
    y_90 = np.clip(
        alpha * y + (1 - alpha) * (last_glucose > 0.5).astype(float)
        + rng2.normal(0, 0.05, N), 0, 1
    ).round().astype(np.float32)

    y_3    = np.stack([y_7, y_30, y_90], axis=1).astype(np.float32)
    users  = np.arange(N, dtype=np.int64)

    return X_base, X_interv, y_3, users


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.uci_diabetes_loader import load_uci_diabetes

    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, ids = load_uci_diabetes()

    Xb, Xi, y3, u = generate_uci_temporal(X_tr, y_tr)
    print("Baseline shape:", Xb.shape)
    print("Intervention shape:", Xi.shape)
    print("Labels shape:", y3.shape)
    print("Label prevalence 7/30/90:", y3[:,0].mean().round(3),
          y3[:,1].mean().round(3), y3[:,2].mean().round(3))

    # Verify intervention effect
    print("\nIntervention activity delta (feature 3) over time:")
    for t in [0, 5, 10, 15, 20, 25, 29]:
        delta = (Xi[:, t, 3] - Xb[:, t, 3]).mean()
        print(f"  t={t:2d}  activity delta = {delta:+.4f}")

    print("\nIntervention glucose delta (feature 0) over time:")
    for t in [0, 5, 10, 15, 20, 25, 29]:
        delta = (Xi[:, t, 0] - Xb[:, t, 0]).mean()
        print(f"  t={t:2d}  glucose delta  = {delta:+.5f}")
