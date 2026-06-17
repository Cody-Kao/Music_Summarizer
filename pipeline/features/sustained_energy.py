# pipeline/features/climax.py

import numpy as np
import librosa


def sustained_energy_score(y, sr):

    """
    Measure sustained loudness / fullness.

    Chorus and EDM drops usually:
    - high RMS
    - relatively stable
    - sustained energy

    Returns:
        {
            "sustained_energy": float,
            "rms_std": float
        }
    """

    rms = librosa.feature.rms(
        y=y
    )[0]

    rms_mean = np.mean(rms)

    rms_std = np.std(rms)

    # ----------------------------------------
    # Stable + high energy
    # ----------------------------------------

    score = rms_mean / (
        rms_std + 1e-6
    )

    return {
        "sustained_energy": float(score),
        "rms_std": float(rms_std)
    }