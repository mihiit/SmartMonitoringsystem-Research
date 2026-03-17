"""
train_lstm.py
=============
Trains the LSTM model with:
  - Proper train / val / test split (stratified)
  - Per-epoch validation loss monitoring
  - AUC reported on held-out TEST set only (prevents leakage)
  - Saves model + training log
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from models.lstm_model import LSTMModel
from utils.preprocessing import load_pima
from utils.clinical_temporal import generate_clinical_temporal

# ── Data ─────────────────────────────────────────────────────────────────────

X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_pima()

# Generate longitudinal trajectories for each split
X_tr_seq, _, y_tr_seq, u_tr = generate_clinical_temporal(X_train, y_train)
X_va_seq, _, y_va_seq, u_va = generate_clinical_temporal(X_val,   y_val)
X_te_seq, _, y_te_seq, u_te = generate_clinical_temporal(X_test,  y_test)

# Convert to tensors
def to_t(arr):    return torch.tensor(arr).float()
def to_ti(arr):   return torch.tensor(arr).long()

X_tr = to_t(X_tr_seq);  y_tr = to_t(y_tr_seq);  u_tr = to_ti(u_tr)
X_va = to_t(X_va_seq);  y_va = to_t(y_va_seq);  u_va = to_ti(u_va)
X_te = to_t(X_te_seq);  y_te = to_t(y_te_seq);  u_te = to_ti(u_te)

# ── Model ─────────────────────────────────────────────────────────────────────

num_users = len(u_tr)
model = LSTMModel(input_size=6, hidden_size=64, num_layers=2,
                  num_users=num_users, dropout=0.20)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCELoss()

EPOCHS = 50
log = []

print("\n=== Training LSTM ===")
for epoch in range(EPOCHS):
    model.train()
    pred_tr = model(X_tr, u_tr)
    loss_tr = criterion(pred_tr, y_tr)
    optimizer.zero_grad()
    loss_tr.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_va = model(X_va, u_va)
        loss_va = criterion(pred_va, y_va)

    log.append({"epoch": epoch+1, "train_loss": loss_tr.item(),
                 "val_loss": loss_va.item()})

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | train_loss={loss_tr.item():.4f} "
              f"| val_loss={loss_va.item():.4f}")

# ── Test evaluation (AUC on held-out test set) ────────────────────────────────

model.eval()
with torch.no_grad():
    pred_te = model(X_te, u_te)

# Use output index 0 (7-day risk) as primary metric, consistent across all evals
auc = roc_auc_score(y_te[:, 0].numpy(), pred_te[:, 0].numpy())
print(f"\nLSTM Test AUC (7-day, held-out): {auc:.4f}")

# Save
torch.save(model.state_dict(), "lstm_model.pth")
print("Model saved: lstm_model.pth")
