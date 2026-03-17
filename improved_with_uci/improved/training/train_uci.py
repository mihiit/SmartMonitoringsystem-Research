"""
train_uci.py
============
Trains all three model variants on the REAL UCI Diabetes dataset.

This provides the real-data validation experiment for the paper.
Input dim = 6 (glucose, CV, insulin, activity, meal, hypoglycaemia).
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from utils.uci_diabetes_loader import load_uci_diabetes
from utils.uci_temporal import generate_uci_temporal
from utils.metrics import auc_with_ci, expected_calibration_error

UCI_INPUT_DIM = 6

def to_t(a):  return torch.tensor(a).float()
def to_ti(a): return torch.tensor(a).long()


def train_one(model, Xtr, ytr, utr, Xva, yva, uva,
              lr=0.001, epochs=50, name=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()
    for epoch in range(epochs):
        model.train()
        pred = model(Xtr, utr)
        loss = criterion(pred, ytr)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = criterion(model(Xva, uva), yva).item()
            print(f"  [{name}] Epoch {epoch+1:3d} | train={loss.item():.4f} | val={vl:.4f}")
    return model


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  Training on UCI Diabetes (REAL longitudinal data, N=70)")
    print("="*60)

    # ── Data ─────────────────────────────────────────────────────────────────
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, ids = load_uci_diabetes()

    Xtr_b, Xtr_i, ytr3, utr = generate_uci_temporal(X_tr, y_tr)
    Xva_b, Xva_i, yva3, uva = generate_uci_temporal(X_va, y_va)
    Xte_b, Xte_i, yte3, ute = generate_uci_temporal(X_te, y_te)

    Xtr_t = to_t(Xtr_b); ytr_t = to_t(ytr3); utr_t = to_ti(utr)
    Xva_t = to_t(Xva_b); yva_t = to_t(yva3); uva_t = to_ti(uva)
    Xte_t = to_t(Xte_b); yte_t = to_t(yte3); ute_t = to_ti(ute)

    num_users = len(utr)
    results   = {}

    # ── LSTM ─────────────────────────────────────────────────────────────────
    lstm = LSTMModel(input_size=UCI_INPUT_DIM, hidden_size=32,
                     num_layers=2, num_users=num_users, dropout=0.20)
    lstm = train_one(lstm, Xtr_t, ytr_t, utr_t, Xva_t, yva_t, uva_t,
                     lr=0.001, epochs=50, name="LSTM")
    lstm.eval()
    with torch.no_grad(): p = lstm(Xte_t, ute_t)
    auc, lo, hi = auc_with_ci(yte_t[:,0].numpy(), p[:,0].numpy(), n_boot=200)
    ece = expected_calibration_error(yte_t[:,0].numpy(), p[:,0].numpy())
    results["LSTM"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(lstm.state_dict(), "uci_lstm.pth")

    # ── Transformer (personalised) ────────────────────────────────────────────
    trans = TransformerModel(input_dim=UCI_INPUT_DIM, num_users=num_users,
                              use_personalization=True, dropout=0.20)
    trans = train_one(trans, Xtr_t, ytr_t, utr_t, Xva_t, yva_t, uva_t,
                      lr=0.0005, epochs=50, name="Transformer")
    trans.eval()
    with torch.no_grad(): p = trans(Xte_t, ute_t)
    auc, lo, hi = auc_with_ci(yte_t[:,0].numpy(), p[:,0].numpy(), n_boot=200)
    ece = expected_calibration_error(yte_t[:,0].numpy(), p[:,0].numpy())
    results["Transformer (Personalised)"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(trans.state_dict(), "uci_transformer.pth")

    # ── Transformer (no personalisation) ─────────────────────────────────────
    trans_nop = TransformerModel(input_dim=UCI_INPUT_DIM, num_users=num_users,
                                  use_personalization=False, dropout=0.20)
    trans_nop = train_one(trans_nop, Xtr_t, ytr_t, utr_t, Xva_t, yva_t, uva_t,
                           lr=0.0005, epochs=50, name="Trans-NoPers")
    trans_nop.eval()
    with torch.no_grad(): p = trans_nop(Xte_t, ute_t)
    auc, lo, hi = auc_with_ci(yte_t[:,0].numpy(), p[:,0].numpy(), n_boot=200)
    ece = expected_calibration_error(yte_t[:,0].numpy(), p[:,0].numpy())
    results["Transformer (No Pers.)"] = {"AUC": auc, "CI_lo": lo, "CI_hi": hi, "ECE": ece}
    torch.save(trans_nop.state_dict(), "uci_transformer_nop.pth")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  UCI Results (TEST SET, 7-day risk, N=11)")
    print("="*60)
    print(f"  {'Model':<38} {'AUC':>6}  {'95% CI':>16}  {'ECE':>6}")
    print(f"  {'-'*72}")
    for name, r in results.items():
        print(f"  {name:<38} {r['AUC']:.4f}  "
              f"[{r['CI_lo']:.4f},{r['CI_hi']:.4f}]  {r['ECE']:.4f}")
