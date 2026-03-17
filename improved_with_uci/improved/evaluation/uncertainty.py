"""
uncertainty.py
==============
Proper MC-Dropout uncertainty estimation.

FIX over original:
  - Original called model.train() which only partially activates dropout
    inside TransformerEncoderLayer, giving near-zero variance.
  - New version uses model.enable_mc_dropout() which explicitly sets ALL
    Dropout modules to train mode, while keeping BN/LayerNorm in eval mode.
  - Returns per-patient mean, std, CoV, and a confidence table.
"""

import numpy as np
from utils.metrics import uncertainty_summary


def predict_with_uncertainty(model, X, users, n_runs: int = 50):
    """
    MC-Dropout uncertainty estimation.

    Parameters
    ----------
    model  : model with enable_mc_dropout() method
    X      : (N, T, F) tensor
    users  : (N,) tensor
    n_runs : number of stochastic forward passes

    Returns
    -------
    mean_pred : (N, output_dim)
    std_pred  : (N, output_dim)
    summary   : dict  (scalar uncertainty summary)
    """
    import torch
    model.eval()
    model.enable_mc_dropout()   # ← correct: only dropout stays in train mode

    preds = []
    with torch.no_grad():
        for _ in range(n_runs):
            p = model(X, users).numpy()
            preds.append(p)

    preds = np.array(preds)        # (n_runs, N, output_dim)
    mean  = preds.mean(axis=0)     # (N, output_dim)
    std   = preds.std(axis=0)      # (N, output_dim)

    model.eval()                   # reset all layers back to eval

    summary = uncertainty_summary(mean, std)
    return mean, std, summary


def print_uncertainty_report(mean, std, summary, n_patients: int = 5):
    print("\n─── Predictive Uncertainty Report ──────────────────────")
    print(f"  Mean risk (population)  : {summary['mean_risk']:.4f}")
    print(f"  Mean σ   (population)  : {summary['mean_std']:.4f}")
    print(f"  Confidence              : {summary['confidence']*100:.2f} %")
    print(f"  CoV                     : {summary['CoV_%']:.2f} %")
    print(f"\n  Per-patient sample (first {n_patients}):")
    print(f"  {'Patient':>8}  {'7-day risk':>12}  {'σ':>8}  {'CoV %':>8}")
    for i in range(min(n_patients, len(mean))):
        mu_i  = mean[i, 0]
        sig_i = std[i, 0]
        cov_i = sig_i / (mu_i + 1e-9) * 100
        print(f"  {i:>8}  {mu_i:>12.4f}  {sig_i:>8.4f}  {cov_i:>7.2f}%")
    print("──────────────────────────────────────────────────────────\n")
