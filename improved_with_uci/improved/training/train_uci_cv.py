"""
train_uci_cv.py
===============
5-fold stratified cross-validation on UCI Diabetes (N=70).

With only 70 patients, a single 85/15 split gives N=11 test patients
which is too small for reliable AUC estimation (CI spans 0.6 wide).
5-fold CV uses 14 test patients per fold and averages across all 70,
giving much tighter confidence intervals.

Reports:
  - Per-fold AUC, ECE
  - Mean ± std across folds
  - Pooled BR and intervention sensitivity across all test patients
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.uci_diabetes_loader import load_uci_full
from utils.uci_temporal import generate_uci_temporal
from utils.metrics import (auc_with_ci, expected_calibration_error,
                            behavioural_robustness, intervention_significance,
                            uncertainty_summary)
from evaluation.uncertainty import predict_with_uncertainty

UCI_INPUT_DIM = 6
N_FOLDS  = 5
EPOCHS   = 60
SEED     = 42

def to_t(a):  return torch.tensor(np.array(a)).float()
def to_ti(a): return torch.tensor(np.array(a)).long()


def train_model(model, Xtr, ytr, utr, Xva, yva, uva, lr, epochs, name=""):
    opt  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = torch.nn.BCELoss()
    best_val, best_state = 999, None
    for epoch in range(epochs):
        model.train()
        p    = model(Xtr, utr)
        loss = crit(p, ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = crit(model(Xva, uva), yva).item()
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    # Restore best checkpoint
    model.load_state_dict(best_state)
    return model


def run_cv(model_class, model_kwargs, lr, name):
    X_all, y_all, _ = load_uci_full()
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_aucs, fold_eces = [], []
    all_R_b, all_R_i, all_y = [], [], []

    print(f"\n  --- {name} ---")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
        X_tr_raw = X_all[tr_idx]; y_tr = y_all[tr_idx]
        X_te_raw = X_all[te_idx]; y_te = y_all[te_idx]

        # Scale per fold (fit on train only)
        N_tr, T, F = X_tr_raw.shape
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw.reshape(-1, F)).reshape(N_tr, T, F)
        X_te = scaler.transform(X_te_raw.reshape(-1, F)).reshape(len(te_idx), T, F)

        # Build a small validation set from training fold (last 10%)
        n_val = max(2, int(len(tr_idx) * 0.12))
        X_va = X_tr[-n_val:]; y_va = y_tr[-n_val:]
        X_tr = X_tr[:-n_val]; y_tr_t = y_tr[:-n_val]

        # Generate temporal branches
        Xtr_b, Xtr_i, ytr3, utr = generate_uci_temporal(X_tr, y_tr_t)
        Xva_b, _,     yva3, uva = generate_uci_temporal(X_va, y_va)
        Xte_b, Xte_i, yte3, ute = generate_uci_temporal(X_te, y_te)

        Xtr_t = to_t(Xtr_b); ytr_t2 = to_t(ytr3); utr_t = to_ti(utr)
        Xva_t = to_t(Xva_b); yva_t2 = to_t(yva3); uva_t = to_ti(uva)
        Xte_t = to_t(Xte_b); Xte_it = to_t(Xte_i); ute_t = to_ti(ute)

        num_u = len(utr)
        kw    = {**model_kwargs, "num_users": num_u}
        model = model_class(**kw)
        model = train_model(model, Xtr_t, ytr_t2, utr_t,
                            Xva_t, yva_t2, uva_t, lr=lr, epochs=EPOCHS)

        model.eval()
        with torch.no_grad():
            R_b = model(Xte_t,  ute_t).numpy()
            R_i = model(Xte_it, ute_t).numpy()

        y_true = yte3[:, 0]
        auc, lo, hi = auc_with_ci(y_true, R_b[:, 0], n_boot=500)
        ece = expected_calibration_error(y_true, R_b[:, 0])

        fold_aucs.append(auc); fold_eces.append(ece)
        all_R_b.append(R_b[:, 0]); all_R_i.append(R_i[:, 0])
        all_y.append(y_true)

        print(f"    Fold {fold+1}: AUC={auc:.4f} [{lo:.3f},{hi:.3f}]  ECE={ece:.4f}  "
              f"n_test={len(y_te)}")

    # Aggregate
    mean_auc = float(np.mean(fold_aucs))
    std_auc  = float(np.std(fold_aucs))
    mean_ece = float(np.mean(fold_eces))

    # Pool all test patients for BR and Wilcoxon
    R_b_pool = np.concatenate(all_R_b)
    R_i_pool = np.concatenate(all_R_i)
    y_pool   = np.concatenate(all_y)

    auc_pool, lo_pool, hi_pool = auc_with_ci(y_pool, R_b_pool, n_boot=1000)
    br  = behavioural_robustness(R_b_pool, R_i_pool)
    sig = intervention_significance(R_b_pool, R_i_pool)

    print(f"\n    Mean AUC:  {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"    Pooled AUC:{auc_pool:.4f}  [{lo_pool:.4f}, {hi_pool:.4f}]")
    print(f"    Mean ECE:  {mean_ece:.4f}")
    print(f"    BR:        {br['BR']:.6f}")
    print(f"    BR_v:      {br['BR_velocity']:.6f}")
    print(f"    Wilcoxon p:{sig['p_value']:.4e}")
    print(f"    Mean ΔR:   {sig['mean_delta']:+.6f}  ({sig['relative_change_%']:+.3f}%)")

    return {
        "name": name,
        "mean_auc": mean_auc, "std_auc": std_auc,
        "pooled_auc": auc_pool, "CI_lo": lo_pool, "CI_hi": hi_pool,
        "mean_ece": mean_ece,
        "BR": br['BR'], "BR_velocity": br['BR_velocity'],
        "p_value": sig['p_value'],
        "mean_delta": sig['mean_delta'],
        "relative_%": sig['relative_change_%'],
    }


if __name__ == "__main__":
    print("\n" + "="*65)
    print("  UCI Diabetes — 5-Fold Cross-Validation (N=70 real patients)")
    print("="*65)

    results = []

    # LSTM
    r = run_cv(
        LSTMModel,
        {"input_size": UCI_INPUT_DIM, "hidden_size": 32,
         "num_layers": 2, "dropout": 0.20},
        lr=0.001, name="LSTM"
    )
    results.append(r)

    # Transformer (personalised)
    r = run_cv(
        TransformerModel,
        {"input_dim": UCI_INPUT_DIM, "use_personalization": True,
         "dropout": 0.20},
        lr=0.0005, name="Transformer (Personalised)"
    )
    results.append(r)

    # Transformer (no personalisation)
    r = run_cv(
        TransformerModel,
        {"input_dim": UCI_INPUT_DIM, "use_personalization": False,
         "dropout": 0.20},
        lr=0.0005, name="Transformer (No Personalisation)"
    )
    results.append(r)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  FINAL SUMMARY — UCI Real-Data Cross-Validation")
    print("="*65)
    print(f"\n  {'Model':<35} {'Pooled AUC':>10}  {'95% CI':>16}  {'ECE':>6}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['name']:<35} {r['pooled_auc']:>10.4f}  "
              f"[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]  {r['mean_ece']:>6.4f}")

    print(f"\n  {'Model':<35} {'BR':>10}  {'BR_v':>10}  {'Wilcoxon p':>14}  {'ΔR%':>8}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['name']:<35} {r['BR']:>10.5f}  {r['BR_velocity']:>10.5f}  "
              f"{r['p_value']:>14.4e}  {r['relative_%']:>+8.3f}%")

    print()