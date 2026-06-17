# pipeline/features/harmony.py

import librosa
import numpy as np


def extract_chroma(
    y,
    sr
):

    chroma = librosa.feature.chroma_cqt(
        y=y,
        sr=sr
    )

    chroma_mean = np.mean(
        chroma,
        axis=1
    )

    chroma_mean /= (
        np.linalg.norm(chroma_mean)
        + 1e-8
    )

    return chroma_mean