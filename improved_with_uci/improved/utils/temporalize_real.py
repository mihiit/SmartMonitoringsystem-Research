"""
temporalize_real.py
===================
Utility to temporalize real (non-Pima) datasets by adding structured noise.
For use when a future longitudinal dataset is integrated.
"""

import numpy as np


def temporalize(features, targets, seq_len: int = 30,
                noise_std: float = 0.02, random_seed: int = 42):
    """
    Creates a simple noisy temporal extension of static features.
    Suitable as a baseline / sanity check, not for clinical experiments.
    """
    rng = np.random.default_rng(random_seed)
    sequences, labels, user_ids = [], [], []

    for i in range(len(features)):
        base = features[i]
        seq  = [base + rng.normal(0, noise_std, len(base)) for _ in range(seq_len)]
        sequences.append(seq)
        labels.append([targets[i], targets[i], targets[i]])
        user_ids.append(i)

    return (np.array(sequences, dtype=np.float32),
            np.array(labels,    dtype=np.float32),
            np.array(user_ids,  dtype=np.int64))
