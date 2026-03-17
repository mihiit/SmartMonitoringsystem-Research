"""
uci_diabetes_loader.py
======================
Loads and processes the UCI Diabetes dataset (70 real IDDM patients).

Dataset: Kahn (1994), AAAI Spring Symposium on AI in Medicine.
70 patients, weeks-to-months of timestamped glucose, insulin,
exercise, and meal data per patient.

Clinical label definition:
  Outcome = 1  if patient mean glucose > 154 mg/dL (poor glycaemic control)
  Outcome = 0  if patient mean glucose <= 154 mg/dL (good glycaemic control)

  Threshold of 154 mg/dL corresponds to an estimated HbA1c ~7.0%
  (ADA target for diabetes management), computed via the
  Nathan et al. (2008) formula: HbA1c = (mean_glucose + 46.7) / 28.7

Features per day (6):
  0  mean_glucose      Daily mean blood glucose (mg/dL), normalised
  1  glucose_cv        Daily coefficient of variation of glucose
  2  insulin_dose      Total daily insulin units
  3  activity_index    Exercise activity index: 0=none, 0.5=typical, 1=more
  4  meal_index        Meal behaviour index: 0=none, 0.5=typical, 1=more
  5  hypoglycemia_flag Fraction of readings < 70 mg/dL that day

Each patient is represented as a T=30 day sequence using
the first 30 days of available data (or padded if shorter).
"""

import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ── Code mappings ────────────────────────────────────────────────────────────
GLUCOSE_CODES  = {48, 57, 58, 59, 60, 61, 62, 63, 64}
EXERCISE_CODES = {69: 0.5, 70: 1.0, 71: 0.0}   # typical / more / less
MEAL_CODES     = {66: 0.5, 67: 1.0, 68: 0.0}   # typical / more / less
INSULIN_CODES  = {33, 34, 35}
HYPO_THRESHOLD = 70.0   # mg/dL
MEAN_GLUCOSE_THRESHOLD = 154.0   # mg/dL → HbA1c ~7.0%

DATA_DIR = os.path.join(os.path.dirname(__file__),
                        "../data/raw/uci_diabetes")


def _parse_patient_file(filepath: str) -> dict:
    """Parse one patient file into a dict keyed by date string."""
    daily = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
            date_str, _, code_str, value_str = parts[0], parts[1], parts[2], parts[3]
            try:
                code = int(code_str)
                val  = float(value_str)
            except ValueError:
                continue

            if date_str not in daily:
                daily[date_str] = {
                    'glucose': [], 'insulin': 0.0,
                    'activity': [], 'meal': [], 'hypo': 0
                }
            d = daily[date_str]

            if code in GLUCOSE_CODES and val > 0:
                d['glucose'].append(val)
                if val < HYPO_THRESHOLD:
                    d['hypo'] += 1

            if code in INSULIN_CODES and val > 0:
                d['insulin'] += val

            if code in EXERCISE_CODES:
                d['activity'].append(EXERCISE_CODES[code])

            if code in MEAL_CODES:
                d['meal'].append(MEAL_CODES[code])

    return daily


def _daily_features(day: dict, global_mean: float, global_std: float) -> np.ndarray:
    """Convert one day's raw events into a 6-dimensional feature vector."""
    glucose = day['glucose']
    if glucose:
        mean_g  = float(np.mean(glucose))
        cv_g    = float(np.std(glucose) / (mean_g + 1e-6))
        hypo_f  = day['hypo'] / len(glucose)
    else:
        mean_g  = global_mean
        cv_g    = 0.0
        hypo_f  = 0.0

    norm_g  = (mean_g - global_mean) / (global_std + 1e-6)
    insulin = min(day['insulin'] / 100.0, 3.0)   # rough normalisation
    act     = float(np.mean(day['activity'])) if day['activity'] else 0.0
    meal    = float(np.mean(day['meal']))     if day['meal']     else 0.0

    return np.array([norm_g, cv_g, insulin, act, meal, hypo_f], dtype=np.float32)


def load_uci_diabetes(
    data_dir: str = None,
    seq_len:  int = 30,
    random_state: int = 42,
    val_size:  float = 0.10,
    test_size: float = 0.15,
):
    """
    Load all 70 UCI patient files and build T=30 temporal sequences.

    Returns
    -------
    X_train, X_val, X_test  : (N, T, 6) float32 arrays
    y_train, y_val, y_test  : (N,)       int32   outcome labels
    scaler                  : fitted StandardScaler (on train features)
    patient_ids             : list of patient filenames
    """
    if data_dir is None:
        data_dir = DATA_DIR

    # ── Collect all patient glucose values for global normalisation ──────────
    all_glucose = []
    patient_files = sorted(
        f for f in os.listdir(data_dir) if f.startswith('data-')
    )

    raw_patients = {}
    for fname in patient_files:
        fpath = os.path.join(data_dir, fname)
        daily = _parse_patient_file(fpath)
        raw_patients[fname] = daily
        for d in daily.values():
            all_glucose.extend(d['glucose'])

    global_mean = float(np.mean(all_glucose))
    global_std  = float(np.std(all_glucose))

    # ── Build sequences and labels ────────────────────────────────────────────
    sequences = []
    labels    = []
    ids       = []

    for fname, daily in raw_patients.items():
        # Sort days chronologically
        def _parse_date(s):
            try:
                return datetime.strptime(s, '%m-%d-%Y')
            except Exception:
                return datetime(1900, 1, 1)

        sorted_days = sorted(daily.items(), key=lambda x: _parse_date(x[0]))

        # Build day-level feature matrix
        day_features = [_daily_features(d, global_mean, global_std)
                        for _, d in sorted_days]

        # Trim or pad to seq_len
        if len(day_features) >= seq_len:
            seq = day_features[:seq_len]
        else:
            # Pad with last observation (forward-fill)
            pad_len = seq_len - len(day_features)
            seq = day_features + [day_features[-1]] * pad_len

        seq_array = np.stack(seq, axis=0)  # (T, 6)

        # Label: mean glucose > threshold → poor control → 1
        all_g = [g for d in daily.values() for g in d['glucose']]
        label = 1 if np.mean(all_g) > MEAN_GLUCOSE_THRESHOLD else 0

        sequences.append(seq_array)
        labels.append(label)
        ids.append(fname)

    X = np.array(sequences, dtype=np.float32)  # (70, T, 6)
    y = np.array(labels,    dtype=np.int32)

    print(f"UCI Diabetes: {len(X)} patients × {seq_len} days × 6 features")
    print(f"Outcome distribution: {y.sum()} poor control, {(1-y).sum()} good control")
    print(f"Global glucose mean: {global_mean:.1f} mg/dL, std: {global_std:.1f}")

    # ── Stratified split ──────────────────────────────────────────────────────
    X_tv, X_test, y_tv, y_test, id_tv, id_test = train_test_split(
        X, y, ids, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )

    # Scale on first feature dimension only (glucose) — others already 0–1
    scaler = StandardScaler()
    # Flatten → scale → reshape
    n_tr, T, F = X_train.shape
    X_train_s = scaler.fit_transform(X_train.reshape(-1, F)).reshape(n_tr, T, F)
    X_val_s   = scaler.transform(X_val.reshape(-1, X_val.shape[2])).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, X_test.shape[2])).reshape(X_test.shape)

    print(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    return X_train_s, X_val_s, X_test_s, y_train, y_val, y_test, scaler, ids


def load_uci_full(data_dir: str = None):
    """Return all 70 patients as (X, y, ids) without splitting — for analysis."""
    if data_dir is None:
        data_dir = DATA_DIR

    all_glucose_vals = []
    patient_files = sorted(f for f in os.listdir(data_dir) if f.startswith('data-'))
    raw = {}
    for fname in patient_files:
        daily = _parse_patient_file(os.path.join(data_dir, fname))
        raw[fname] = daily
        for d in daily.values():
            all_glucose_vals.extend(d['glucose'])

    gm = float(np.mean(all_glucose_vals))
    gs = float(np.std(all_glucose_vals))

    seqs, labels, ids = [], [], []
    for fname, daily in raw.items():
        def _pd(s):
            try: return datetime.strptime(s, '%m-%d-%Y')
            except: return datetime(1900, 1, 1)
        sorted_days = sorted(daily.items(), key=lambda x: _pd(x[0]))
        day_feat = [_daily_features(d, gm, gs) for _, d in sorted_days]
        seq = day_feat[:30] if len(day_feat) >= 30 else day_feat + [day_feat[-1]] * (30 - len(day_feat))
        all_g = [g for d in daily.values() for g in d['glucose']]
        seqs.append(np.stack(seq))
        labels.append(1 if np.mean(all_g) > MEAN_GLUCOSE_THRESHOLD else 0)
        ids.append(fname)

    return np.array(seqs, dtype=np.float32), np.array(labels, dtype=np.int32), ids


if __name__ == "__main__":
    X_tr, X_va, X_te, y_tr, y_va, y_te, scaler, ids = load_uci_diabetes()
    print("\nFeature shapes:", X_tr.shape, X_va.shape, X_te.shape)
    print("Sample sequence (patient 0, first 5 days):")
    print("  [norm_glucose, glucose_CV, insulin, activity, meal, hypo_flag]")
    for t in range(5):
        print(f"  Day {t+1}: {X_tr[0, t].round(3).tolist()}")
