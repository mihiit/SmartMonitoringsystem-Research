"""
train_nhanes.py
===============
Trains all three model variants on the NHANES-calibrated dataset.
Input size is 8 (vs 6 for Pima).

Run this AFTER train_lstm.py / train_transformer.py (Pima models).
Provides cross-dataset validation as required by top-tier journals.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.nhanes_synthetic import load_nhanes_synthetic
from utils.nhanes_temporal import generate_nhanes_temporal
from utils.metrics import auc_with_ci, expected_calibration_error

NHANES_INPUT_DIM = 8

def to_t(a):  return torch.tensor(a).float()
def to_ti(a): return torch.tensor(a).long()


def train_one(model, X_tr, y_tr, u_tr, X_va, y_va, u_va,
              lr: float = 0.0005, epochs: int = 50, name: str = ""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X_tr, u_tr)
        loss = criterion(pred, y_tr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model(X_va, u_va), y_va).item()
            print(f"  [{name}] Epoch {epoch+1:3d} | train={loss.item():.4f} | val={val_loss:.4f}")

    return model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Training on NHANES-calibrated dataset (8 features)")
    print("="*60)

    # ── Data ──────────────────────────────────────────────────────────
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_nhanes_synthetic()

    X_tr_b, X_tr_i, y_tr, u_tr = generate_nhanes_temporal(X_train, y_train)
    X_va_b, X_va_i, y_va, u_va = generate_nhanes_temporal(X_val,   y_val)
    X_te_b, X_te_i, y_te, u_te = generate_nhanes_temporal(X_test,  y_test)

    Xtr = to_t(X_tr_b); ytr = to_t(y_tr); utr = to_ti(u_tr)
    Xva = to_t(X_va_b); yva = to_t(y_va); uva = to_ti(u_va)
    Xte = to_t(X_te_b); yte = to_t(y_te); ute = to_ti(u_te)

    num_users = len(u_tr)
    results   = {}

    # ── LSTM ──────────────────────────────────────────────────────────
    lstm = LSTMModel(input_size=NHANES_INPUT_DIM, hidden_size=64,
                     num_layers=2, num_users=num_users, dropout=0.20)
    lstm = train_one(lstm, Xtr, ytr, utr, Xva, yva, uva,
                     lr=0.001, epochs=50, name="LSTM")
    lstm.eval()
    with torch.no_grad():
        p = lstm(Xte, ute)
    auc, lo, hi = auc_with_ci(yte[:, 0].numpy(), p[:, 0].numpy())
    ece = expected_calibration_error(yte[:, 0].numpy(), p[:, 0].numpy())
    results["LSTM"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(lstm.state_dict(), "nhanes_lstm.pth")

    # ── Transformer (full) ────────────────────────────────────────────
    trans = TransformerModel(input_dim=NHANES_INPUT_DIM, num_users=num_users,
                              use_personalization=True, dropout=0.20)
    trans = train_one(trans, Xtr, ytr, utr, Xva, yva, uva,
                      lr=0.0005, epochs=50, name="Transformer")
    trans.eval()
    with torch.no_grad():
        p = trans(Xte, ute)
    auc, lo, hi = auc_with_ci(yte[:, 0].numpy(), p[:, 0].numpy())
    ece = expected_calibration_error(yte[:, 0].numpy(), p[:, 0].numpy())
    results["Transformer (Personalized)"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(trans.state_dict(), "nhanes_transformer.pth")

    # ── Transformer (no personalization) ─────────────────────────────
    trans_nop = TransformerModel(input_dim=NHANES_INPUT_DIM, num_users=num_users,
                                  use_personalization=False, dropout=0.20)
    trans_nop = train_one(trans_nop, Xtr, ytr, utr, Xva, yva, uva,
                           lr=0.0005, epochs=50, name="Trans-NoPerson")
    trans_nop.eval()
    with torch.no_grad():
        p = trans_nop(Xte, ute)
    auc, lo, hi = auc_with_ci(yte[:, 0].numpy(), p[:, 0].numpy())
    ece = expected_calibration_error(yte[:, 0].numpy(), p[:, 0].numpy())
    results["Transformer (No Personalization)"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(trans_nop.state_dict(), "nhanes_transformer_nop.pth")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  NHANES Results Summary (TEST SET, 7-day risk)")
    print("="*60)
    print(f"  {'Model':<38} {'AUC':>6}  {'95% CI':>16}  {'ECE':>6}")
    print(f"  {'-'*70}")
    for name, r in results.items():
        print(f"  {name:<38} {r['AUC']:.4f}  "
              f"[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]  {r['ECE']:.4f}")
