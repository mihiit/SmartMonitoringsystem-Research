"""
clinical_temporal.py
====================
Generates synthetic longitudinal trajectories from the Pima static dataset.

Key improvements over original:
  - Stochastic noise on each step (realistic biological variability)
  - Delayed / lagged intervention via exponential ramp-up (physiological inertia)
  - Returns BOTH baseline and intervention sequences for sensitivity analysis
  - Latent resilience drawn from Beta distribution (realistic inter-patient spread)
  - Richer multi-factor label formula anchored to ground-truth outcomes
  - No more clipping to the same tiny [-2,3] band for all features
"""

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_clinical_temporal(
    features,
    targets,
    seq_len: int = 30,
    noise_std: float = 0.02,
    intervention_start: int = 10,
    intervention_delta: float = 0.30,
    lag_tau: float = 5.0,
    random_seed: int = 42,
):
    """
    Parameters
    ----------
    features          : (N, d) normalised clinical features from Pima dataset
    targets           : (N,)   binary diabetes outcomes
    seq_len           : length of each trajectory
    noise_std         : per-step Gaussian noise (biological variability)
    intervention_start: time step at which lifestyle intervention begins
    intervention_delta: magnitude of activity increase under intervention
    lag_tau           : exponential lag time constant (physiological inertia)
    random_seed       : reproducibility

    Returns
    -------
    X_base   : (N, T, 6)  baseline trajectories
    X_interv : (N, T, 6)  post-intervention trajectories
    y        : (N, 3)     7/30/90-day probabilistic risk labels
    users    : (N,)       integer patient IDs
    """

    rng = np.random.default_rng(random_seed)
    n = len(features)

    # Per-patient latent resilience – Beta(2,5) skews toward lower resilience
    resilience_vec = rng.beta(a=2, b=5, size=n)

    X_base_all   = []
    X_interv_all = []
    labels       = []

    for i in range(n):
        base       = features[i]
        outcome    = int(targets[i])
        resilience = resilience_vec[i]

        age_scaled = float(base[7])
        bmi_scaled = float(base[5])
        g0         = float(base[1])   # glucose
        p0         = float(base[2])   # blood pressure

        age_factor        = 0.01 + age_scaled * 0.002
        resilience_effect = 0.30 - resilience   # positive value → faster progression

        # Running state for each trajectory branch
        g_b, p_b = g0, p0   # baseline
        g_i, p_i = g0, p0   # intervention

        seq_b, seq_i = [], []

        for t in range(seq_len):

            # ── Activity index ──────────────────────────────────────────────
            act_base = np.clip(0.5 - 0.20 * bmi_scaled - 0.003 * t, 0.0, 1.0)

            if t >= intervention_start:
                ramp    = 1.0 - np.exp(-(t - intervention_start) / lag_tau)
                act_int = np.clip(act_base + intervention_delta * ramp, 0.0, 1.0)
            else:
                act_int = act_base

            # ── Metabolic step ───────────────────────────────────────────────
            def _step(g, p, act):
                dg = (age_factor + 0.015 * bmi_scaled + 0.008 * p
                      - 0.040 * act + resilience_effect
                      + rng.normal(0, noise_std))
                dp = (age_factor + 0.010 * bmi_scaled
                      - 0.030 * act + resilience_effect
                      + rng.normal(0, noise_std))
                g_new = np.clip(g + dg, -3.0, 3.0)
                p_new = np.clip(p + dp, -3.0, 3.0)
                return g_new, p_new

            g_b, p_b = _step(g_b, p_b, act_base)
            g_i, p_i = _step(g_i, p_i, act_int)

            def _row(g, p, act):
                cv = np.clip(0.25*p + 0.25*g + 0.25*bmi_scaled + 0.25*age_scaled,
                             -3.0, 3.0)
                return [p, g, bmi_scaled, act, age_factor, cv]

            seq_b.append(_row(g_b, p_b, act_base))
            seq_i.append(_row(g_i, p_i, act_int))

        X_base_all.append(seq_b)
        X_interv_all.append(seq_i)

        # ── Labels: richer multi-factor risk, anchored to ground truth ──────
        final_g  = seq_b[-1][1]
        final_p  = seq_b[-1][0]
        final_cv = seq_b[-1][5]

        base_risk = (0.25 * bmi_scaled + 0.25 * age_scaled
                     + 0.20 * final_g   + 0.15 * final_cv
                     + 0.10 * final_p   + 0.05 * (1.0 - resilience))

        alpha = 0.60   # how strongly to anchor to real label
        p7  = float(np.clip(alpha * outcome + (1-alpha) * _sigmoid(base_risk * 0.8), 0, 1))
        p30 = float(np.clip(alpha * outcome + (1-alpha) * _sigmoid(base_risk * 1.0), 0, 1))
        p90 = float(np.clip(alpha * outcome + (1-alpha) * _sigmoid(base_risk * 1.2), 0, 1))

        labels.append([
            float(rng.random() < p7),
            float(rng.random() < p30),
            float(rng.random() < p90),
        ])

    return (
        np.array(X_base_all,   dtype=np.float32),
        np.array(X_interv_all, dtype=np.float32),
        np.array(labels,       dtype=np.float32),
        np.arange(n,           dtype=np.int64),
    )
