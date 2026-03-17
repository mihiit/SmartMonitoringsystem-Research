"""
preprocessing.py
================
Loads and preprocesses the Pima dataset with a proper train/val/test split.

Improvements over original:
  - Returns stratified train / val / test splits (not just full data)
  - Proper zero-replacement only for clinically invalid columns
  - Exposes the fitted scaler for inverse-transform (needed for interpretability)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Columns where 0 is physiologically impossible
_ZERO_INVALID = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def load_pima(
    path: str = "data/raw/diabetes.csv",
    val_size: float = 0.10,
    test_size: float = 0.15,
    random_state: int = 42,
):
    """
    Returns
    -------
    (features_train, features_val, features_test,
     targets_train,  targets_val,  targets_test,
     scaler)
    """
    df = pd.read_csv(path)

    # Replace physiologically impossible zeros with column median
    for col in _ZERO_INVALID:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    # Stratified split: ensures balanced class ratio in each split
    X_tv, X_test, y_tv, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def load_pima_full(path: str = "data/raw/diabetes.csv"):
    """Convenience loader returning (all_features, all_targets, scaler)."""
    df = pd.read_csv(path)
    for col in _ZERO_INVALID:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    X = df.drop("Outcome", axis=1).values
    y = df["Outcome"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler
