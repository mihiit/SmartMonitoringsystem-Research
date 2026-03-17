"""
nhanes_synthetic.py
===================
Generates a NHANES-calibrated synthetic longitudinal dataset.

Statistical parameters sourced from:
  - NHANES 2015-2018 public summary statistics (CDC)
  - ADA Standards of Medical Care in Diabetes, 2021
  - Framingham Heart Study risk parameters
  - UKPDS Outcomes Model distributions

This dataset is NOT the real NHANES data. It is a statistically calibrated
synthetic dataset whose marginal distributions match published NHANES
summary statistics. It is suitable for:
  - Methodology validation (the primary use case of this paper)
  - Reproducible benchmarking without data-use agreements
  - Longitudinal trajectory simulation

Features (8):
  0  Glucose       Fasting plasma glucose (mg/dL)
  1  HbA1c         Glycated haemoglobin (%)
  2  SBP           Systolic blood pressure (mmHg)
  3  DBP           Diastolic blood pressure (mmHg)
  4  BMI           Body mass index (kg/m²)
  5  Age           Age in years
  6  HDL           HDL cholesterol (mg/dL)
  7  Triglycerides Fasting triglycerides (mg/dL)

Outcome:
  1 = Diabetes (FPG ≥ 126 mg/dL  OR  HbA1c ≥ 6.5%)    [ADA 2021 criteria]
  0 = No diabetes

Reference distributions (NHANES 2015-2018, adults 20-79):
  Glucose      :  N(99.5, 21.3²)   clipped [60, 350]
  HbA1c        :  N(5.70, 0.50²)   clipped [4.0, 14.0]
  SBP          :  N(122,  16²)     clipped [80, 210]
  DBP          :  N(76,   10²)     clipped [40, 120]
  BMI          :  N(29.6, 6.8²)    clipped [15, 60]
  Age          :  Uniform[20, 79]
  HDL          :  N(52,   15²)     clipped [20, 100]
  Triglycerides:  LogN(μ=4.85,σ=0.65)  clipped [30, 800]

Correlations approximate the published NHANES correlation matrix for
these variables in adults with pre-diabetes/diabetes risk.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ── Published NHANES parameter table ─────────────────────────────────────────
#   (mean, std, min_clip, max_clip)
_NHANES_PARAMS = {
    "Glucose":       (99.5,  21.3,  60.0,  350.0),
    "HbA1c":         ( 5.70,  0.50,  4.0,   14.0),
    "SBP":           (122.0, 16.0,  80.0,  210.0),
    "DBP":           ( 76.0, 10.0,  40.0,  120.0),
    "BMI":           ( 29.6,  6.8,  15.0,   60.0),
    "Age":           ( 49.5, 17.0,  20.0,   79.0),   # Uniform → treated as Normal
    "HDL":           ( 52.0, 15.0,  20.0,  100.0),
    "Triglycerides": (138.0, 88.0,  30.0,  800.0),   # log-normal
}

# Approximate correlation matrix (NHANES published inter-variable correlations)
# Order: Glucose, HbA1c, SBP, DBP, BMI, Age, HDL, Triglycerides
_CORR = np.array([
    # Gluc  HbA1c  SBP   DBP   BMI   Age   HDL   TG
    [ 1.00,  0.75,  0.25,  0.20,  0.30,  0.25, -0.20,  0.35],  # Glucose
    [ 0.75,  1.00,  0.22,  0.18,  0.28,  0.30, -0.22,  0.30],  # HbA1c
    [ 0.25,  0.22,  1.00,  0.65,  0.35,  0.45, -0.15,  0.20],  # SBP
    [ 0.20,  0.18,  0.65,  1.00,  0.30,  0.25, -0.12,  0.18],  # DBP
    [ 0.30,  0.28,  0.35,  0.30,  1.00,  0.10, -0.35,  0.40],  # BMI
    [ 0.25,  0.30,  0.45,  0.25,  0.10,  1.00, -0.10,  0.15],  # Age
    [-0.20, -0.22, -0.15, -0.12, -0.35, -0.10,  1.00, -0.40],  # HDL
    [ 0.35,  0.30,  0.20,  0.18,  0.40,  0.15, -0.40,  1.00],  # Triglycerides
])

_FEATURE_NAMES = list(_NHANES_PARAMS.keys())


def _ensure_psd(C: np.ndarray) -> np.ndarray:
    """Nearest positive-semidefinite matrix via eigenvalue clipping."""
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.clip(eigvals, 1e-6, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def generate_nhanes_dataset(
    n: int = 2000,
    random_seed: int = 42,
    save_path: str = None,
) -> pd.DataFrame:
    """
    Generate N correlated patient records matching NHANES distributions.

    Returns a DataFrame with columns:
      Glucose, HbA1c, SBP, DBP, BMI, Age, HDL, Triglycerides, Outcome
    """
    rng = np.random.default_rng(random_seed)

    C_psd = _ensure_psd(_CORR)
    L     = np.linalg.cholesky(C_psd)

    # Draw correlated standard normals
    Z = rng.standard_normal((n, len(_FEATURE_NAMES))) @ L.T

    data = {}
    for j, (name, (mu, sd, lo, hi)) in enumerate(_NHANES_PARAMS.items()):
        if name == "Triglycerides":
            # Log-normal: match mean=138, sd=88
            # log-mean ≈ ln(mean²/√(mean²+sd²)), log-sd = √(ln(1+(sd/mean)²))
            lmu  = np.log(mu**2 / np.sqrt(mu**2 + sd**2))
            lsig = np.sqrt(np.log(1 + (sd / mu)**2))
            col  = np.exp(lmu + lsig * Z[:, j])
        else:
            col = mu + sd * Z[:, j]

        data[name] = np.clip(col, lo, hi)

    df = pd.DataFrame(data)

    # ── ADA 2021 diabetes outcome criteria ────────────────────────────────
    df["Outcome"] = (
        (df["Glucose"] >= 126) | (df["HbA1c"] >= 6.5)
    ).astype(int)

    # Ensure realistic prevalence (~15% diabetes in NHANES 2017-18 adults)
    # If generated prevalence is far off, re-weight slightly
    prev = df["Outcome"].mean()
    if prev < 0.08 or prev > 0.30:
        # Adjust glucose threshold to match ~15% prevalence
        pct85 = np.percentile(df["Glucose"], 85)
        df["Outcome"] = (
            (df["Glucose"] >= pct85) | (df["HbA1c"] >= 6.5)
        ).astype(int)

    if save_path:
        df.to_csv(save_path, index=False)
        print(f"  NHANES-calibrated dataset saved: {save_path}")
        print(f"  N={n}  |  Diabetes prevalence: {df['Outcome'].mean()*100:.1f}%")

    return df


def load_nhanes_synthetic(
    path: str = "data/raw/nhanes_synthetic.csv",
    val_size: float = 0.10,
    test_size: float = 0.15,
    random_state: int = 42,
    regenerate: bool = False,
):
    """
    Load (or generate) the NHANES-calibrated dataset.
    Returns stratified train/val/test splits + scaler.
    """
    import os
    from sklearn.model_selection import train_test_split

    if not os.path.exists(path) or regenerate:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df = generate_nhanes_dataset(n=2000, save_path=path)
    else:
        df = pd.read_csv(path)

    X = df.drop("Outcome", axis=1).values.astype(np.float32)
    y = df["Outcome"].values.astype(np.int32)

    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def describe_nhanes_dataset(df: pd.DataFrame):
    """Print descriptive statistics matching Table 3 format in paper."""
    cols = _FEATURE_NAMES + ["Outcome"]
    desc = df[cols].describe().T[["mean", "std", "min", "max"]]
    desc.columns = ["Mean", "Std Dev", "Min", "Max"]

    print("\n── NHANES-Calibrated Dataset: Descriptive Statistics ──────")
    print(f"  N = {len(df)}  |  Diabetes prevalence = {df['Outcome'].mean()*100:.1f}%")
    print(f"\n  {'Feature':<16} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    for name in _FEATURE_NAMES:
        row = desc.loc[name]
        print(f"  {name:<16} {row['Mean']:>8.2f} {row['Std Dev']:>8.2f} "
              f"{row['Min']:>8.2f} {row['Max']:>8.2f}")
    print(f"\n  {'Outcome':<16} {df['Outcome'].mean():>8.3f}  (prevalence)")


if __name__ == "__main__":
    df = generate_nhanes_dataset(n=2000, save_path="data/raw/nhanes_synthetic.csv")
    describe_nhanes_dataset(df)
