"""
main.py
=======
Master entry point. Runs training for all three model variants,
then runs the full evaluation pipeline.

Usage:
    python main.py

All three models must be trained before evaluation can run.
Training scripts use identical data splits and hyperparameters
to ensure comparability.
"""

import os
import sys

print("\n" + "="*60)
print("  SmartMonitoringSystem — Temporal Disease Progression")
print("="*60)

# ── Step 1: Train LSTM ──────────────────────────────────────────────────────
print("\n[1/4] Training LSTM...")
ret = os.system("python training/train_lstm.py")
if ret != 0:
    print("ERROR: LSTM training failed."); sys.exit(1)

# ── Step 2: Train Transformer (personalized) ────────────────────────────────
print("\n[2/4] Training Transformer (personalized)...")
ret = os.system("python training/train_transformer.py")
if ret != 0:
    print("ERROR: Transformer training failed."); sys.exit(1)

# ── Step 3: Train Transformer (no personalization) ──────────────────────────
print("\n[3/4] Training Transformer (no personalization — ablation)...")
ret = os.system("python training/train_transformer_no_personal.py")
if ret != 0:
    print("ERROR: Ablation model training failed."); sys.exit(1)

# ── Step 4: Full evaluation ──────────────────────────────────────────────────
print("\n[4/4] Running full evaluation pipeline...")
ret = os.system("python evaluation/compare_models.py")
if ret != 0:
    print("ERROR: Evaluation failed."); sys.exit(1)

print("\n" + "="*60)
print("  Pipeline complete. See figures/ for all outputs.")
print("="*60 + "\n")
