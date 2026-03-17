"""
onset.py
========
Time-to-onset estimation from a risk trajectory.
"""

import numpy as np


def estimate_onset(risk_vector: list, threshold: float = 0.70) -> dict:
    """
    Estimate when predicted risk first crosses a threshold.

    Parameters
    ----------
    risk_vector : list/array of risk probabilities at [7, 30, 90] days
    threshold   : risk probability threshold for onset classification

    Returns
    -------
    dict with onset_day, max_risk, and risk_at_each_horizon
    """
    horizons = [7, 30, 90]
    rv       = list(risk_vector)

    onset_day = None
    for day, r in zip(horizons, rv):
        if r >= threshold:
            onset_day = day
            break

    return {
        "onset_day":   onset_day if onset_day is not None else "No onset predicted",
        "max_risk":    float(max(rv)),
        "risk_7day":   float(rv[0]),
        "risk_30day":  float(rv[1]),
        "risk_90day":  float(rv[2]),
        "threshold":   threshold,
    }


def print_onset_report(result: dict):
    print("\n─── Onset Prediction ──────────────────────────────────────")
    print(f"  Risk @ 7-day  : {result['risk_7day']:.4f}")
    print(f"  Risk @ 30-day : {result['risk_30day']:.4f}")
    print(f"  Risk @ 90-day : {result['risk_90day']:.4f}")
    print(f"  Threshold     : {result['threshold']:.2f}")
    print(f"  Onset day     : {result['onset_day']}")
    print("──────────────────────────────────────────────────────────\n")
