"""
metrics.py
==========
All quantitative metrics used in the paper:
  - AUC with 95 % bootstrap CI
  - Calibration error (ECE)
  - Behavioural robustness (BR, BR_velocity)
  - Risk velocity
  - Paired Wilcoxon test for intervention effect
  - Predictive uncertainty (mean ± std, CoV)
"""

import numpy as np
from sklearn.metrics import roc_auc_score


# ─────────────────────────────────────────────────────────────────────────────
# AUC with bootstrap confidence interval
# ─────────────────────────────────────────────────────────────────────────────

def auc_with_ci(y_true, y_score, n_boot: int = 1000, ci: float = 0.95,
                random_state: int = 42):
    """
    Returns (auc, lower_ci, upper_ci).
    Works for multi-output y_score by averaging across outputs.
    """
    rng = np.random.default_rng(random_state)

    if y_score.ndim > 1:
        y_score_1d = y_score.mean(axis=1)
    else:
        y_score_1d = y_score

    if y_true.ndim > 1:
        y_true_1d = y_true[:, 0]
    else:
        y_true_1d = y_true

    base_auc = roc_auc_score(y_true_1d, y_score_1d)

    boot_aucs = []
    n = len(y_true_1d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true_1d[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true_1d[idx], y_score_1d[idx]))

    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_aucs, 100 * alpha))
    hi = float(np.percentile(boot_aucs, 100 * (1 - alpha)))

    return float(base_auc), lo, hi


# ─────────────────────────────────────────────────────────────────────────────
# Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true, y_prob, n_bins: int = 10):
    """
    ECE: weighted mean absolute difference between predicted probability
    and empirical accuracy within equal-width probability bins.
    """
    if y_prob.ndim > 1:
        y_prob = y_prob.mean(axis=1)
    if y_true.ndim > 1:
        y_true = y_true[:, 0]

    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    n      = len(y_true)

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)

    return float(ece)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Velocity
# ─────────────────────────────────────────────────────────────────────────────

def risk_velocity(risk_sequence):
    """
    Compute per-step velocity for a 1-D risk sequence.
    Returns array of length (T-1).
    """
    r = np.asarray(risk_sequence, dtype=float)
    return np.diff(r)          # Vt = R_t - R_{t-1}  (dt = 1)


# ─────────────────────────────────────────────────────────────────────────────
# Behavioural Robustness
# ─────────────────────────────────────────────────────────────────────────────

def behavioural_robustness(R_base, R_interv):
    """
    BR  = mean |R'_t - R_t| over time (eq. 16 in paper)
    BR_v = mean |V'_t - V_t| over time (eq. 17)

    Parameters
    ----------
    R_base   : (T,) or (N, T)
    R_interv : same shape

    Returns dict with 'BR' and 'BR_velocity'.
    """
    R_base   = np.asarray(R_base,   dtype=float)
    R_interv = np.asarray(R_interv, dtype=float)

    diff = np.abs(R_interv - R_base)
    BR   = float(diff.mean())

    if R_base.ndim == 1:
        V_base   = np.diff(R_base)
        V_interv = np.diff(R_interv)
    else:
        V_base   = np.diff(R_base,   axis=-1)
        V_interv = np.diff(R_interv, axis=-1)

    BR_v = float(np.abs(V_interv - V_base).mean())

    return {"BR": BR, "BR_velocity": BR_v}


# ─────────────────────────────────────────────────────────────────────────────
# Intervention effect – Wilcoxon signed-rank test
# ─────────────────────────────────────────────────────────────────────────────

def intervention_significance(R_base, R_interv):
    """
    Paired Wilcoxon signed-rank test on flattened risk differences.
    Returns (statistic, p_value, mean_delta, relative_change_pct).
    """
    from scipy.stats import wilcoxon

    R_base   = np.asarray(R_base,   dtype=float).flatten()
    R_interv = np.asarray(R_interv, dtype=float).flatten()

    delta = R_interv - R_base

    stat, p = wilcoxon(delta, alternative="two-sided", zero_method="wilcox")
    mean_delta  = float(delta.mean())
    rel_change  = float(mean_delta / (R_base.mean() + 1e-9) * 100)

    return {
        "statistic":         float(stat),
        "p_value":           float(p),
        "mean_delta":        mean_delta,
        "relative_change_%": rel_change,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Predictive uncertainty summary
# ─────────────────────────────────────────────────────────────────────────────

def uncertainty_summary(mean_preds, std_preds):
    """
    Returns dict with mean risk, std, confidence, and CoV.
    mean_preds / std_preds: (N, output_dim) or (N,)
    """
    mu  = float(np.mean(mean_preds))
    sig = float(np.mean(std_preds))
    cov = sig / (mu + 1e-9)

    return {
        "mean_risk":   mu,
        "mean_std":    sig,
        "confidence":  float(1.0 - sig),
        "CoV":         float(cov),
        "CoV_%":       float(cov * 100),
    }
