"""
ablation.py
===========
Ablation study: compare LSTM, Transformer (full), Transformer (no personalization).

FIX over original:
  - Original ran roc_auc_score on all 3 outputs averaged, inflating ablation AUC
    to 0.92–0.98 while the main evaluation showed 0.57–0.61.
  - Now consistently uses y[:, 0] (7-day risk) across ALL evaluations,
    matching Table 6 in the paper.
  - Reports AUC with 95% bootstrap CI for each model.
  - Adds a brief narrative explaining why removing personalization may score
    comparably (shared population dynamics).
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from utils.metrics import auc_with_ci


def run_ablation(models: dict, X, y, users, output_idx: int = 0):
    """
    Parameters
    ----------
    models     : dict of {name: model}
    X          : (N, T, F) float tensor
    y          : (N, 3)    float tensor  — labels for 7/30/90-day risk
    users      : (N,)      long tensor
    output_idx : which output to use for AUC (0 = 7-day, default)
    """
    import torch
    y_true = y[:, output_idx].numpy()

    print(f"\n{'─'*58}")
    print(f" Ablation Study  (output idx={output_idx} → "
          f"{'7-day' if output_idx==0 else '30-day' if output_idx==1 else '90-day'} risk)")
    print(f"{'─'*58}")
    print(f"  {'Model':<38} {'AUC':>6}  95% CI")
    print(f"{'─'*58}")

    results = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            pred = model(X, users)

        y_score = pred[:, output_idx].numpy()
        auc, lo, hi = auc_with_ci(y_true, y_score)
        results[name] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi}
        print(f"  {name:<38} {auc:.4f}  [{lo:.4f}, {hi:.4f}]")

    print(f"{'─'*58}")

    # Interpretive note
    best  = max(results, key=lambda k: results[k]["AUC"])
    worst = min(results, key=lambda k: results[k]["AUC"])
    gap   = results[best]["AUC"] - results[worst]["AUC"]
    print(f"\n  Best model : {best}  (AUC {results[best]['AUC']:.4f})")
    print(f"  AUC gap    : {gap:.4f}  "
          + ("(small → shared population dynamics dominate)"
             if gap < 0.02 else
             "(meaningful → personalization / architecture matters)"))

    return results
