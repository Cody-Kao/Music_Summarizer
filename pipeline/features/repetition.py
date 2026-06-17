# pipeline/features/repetition.py

"""
Hook sections often exhibit:

repeated vocal phrases
repeated harmonic loops
repeated rhythmic motifs

This creates: stronger self-similarity.
"""

import librosa
import numpy as np


def repetition_score(
    y,
    sr
):
    """
    Measure internal repetition / hookiness.

    Uses MFCC self-similarity.
    """

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=13
    )

    # -------------------------------
    # cosine self similarity
    # -------------------------------

    mfcc = mfcc.T

    mfcc_norm = mfcc / (
        np.linalg.norm(
            mfcc,
            axis=1,
            keepdims=True
        ) + 1e-8
    )

    sim = np.dot(
        mfcc_norm,
        mfcc_norm.T
    )

    # remove diagonal
    np.fill_diagonal(sim, 0)

    # average repeated similarity
    repetition = np.mean(sim)

    return float(repetition)