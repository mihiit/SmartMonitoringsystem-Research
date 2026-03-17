"""
train_transformer_no_personal.py
=================================
Trains the non-personalized Transformer (ablation: no identity embeddings).
Uses identical hyperparameters and data splits as the personalized variant.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from sklearn.metrics import roc_auc_score

from models.transformer_model import TransformerModel
from utils.preprocessing import load_pima
from utils.clinical_temporal import generate_clinical_temporal

X_train, X_val, X_test, y_train, y_val, y_test, _ = load_pima()

X_tr_seq, _, y_tr_seq, u_tr = generate_clinical_temporal(X_train, y_train)
X_va_seq, _, y_va_seq, u_va = generate_clinical_temporal(X_val,   y_val)
X_te_seq, _, y_te_seq, u_te = generate_clinical_temporal(X_test,  y_test)

def to_t(a):  return torch.tensor(a).float()
def to_ti(a): return torch.tensor(a).long()

X_tr = to_t(X_tr_seq); y_tr = to_t(y_tr_seq); u_tr = to_ti(u_tr)
X_va = to_t(X_va_seq); y_va = to_t(y_va_seq); u_va = to_ti(u_va)
X_te = to_t(X_te_seq); y_te = to_t(y_te_seq); u_te = to_ti(u_te)

# NOTE: use_personalization=False — this is the ablation variant
model = TransformerModel(input_dim=6, num_users=len(u_tr),
                         use_personalization=False, dropout=0.20)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.BCELoss()

EPOCHS = 50

print("\n=== Training Transformer (No Personalization — Ablation) ===")
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

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d} | train={loss_tr.item():.4f} "
              f"| val={loss_va.item():.4f}")

model.eval()
with torch.no_grad():
    pred_te = model(X_te, u_te)

auc = roc_auc_score(y_te[:, 0].numpy(), pred_te[:, 0].numpy())
print(f"\nTransformer (No Personalization) Test AUC (7-day): {auc:.4f}")

torch.save(model.state_dict(), "transformer_no_personal.pth")
print("Model saved: transformer_no_personal.pth")
