"""
dataset_builder.py
==================
Kept for backward compatibility.
For the main experiments, use clinical_temporal.py + preprocessing.py instead.

generate_temporal_data() here produces simple synthetic data with Gaussian noise —
it is NOT clinically grounded and should not be used for paper experiments.
"""

import numpy as np


def generate_temporal_data(num_users: int = 100, seq_len: int = 30,
                            random_seed: int = 42):
    """
    Simple synthetic dataset for quick unit-testing of model architectures.
    NOT suitable for clinical experiments — use generate_clinical_temporal() instead.
    """
    rng = np.random.default_rng(random_seed)

    data, labels, user_ids = [], [], []

    for user in range(num_users):
        base_bp      = rng.normal(120, 10)
        base_glucose = rng.normal(90, 10)

        seq = []
        for t in range(seq_len):
            bp      = base_bp      + rng.normal(0, 5) + t * 0.2
            glucose = base_glucose + rng.normal(0, 5) + t * 0.3
            bmi     = rng.normal(25, 3)
            hr      = rng.normal(70, 5)
            steps   = rng.normal(5000, 1500)
            seq.append([bp, glucose, bmi, hr, steps])

        risk = 1 if base_glucose > 130 or base_bp > 140 else 0

        data.append(seq)
        labels.append([risk, risk, risk])
        user_ids.append(user)

    return (np.array(data, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(user_ids, dtype=np.int64))
