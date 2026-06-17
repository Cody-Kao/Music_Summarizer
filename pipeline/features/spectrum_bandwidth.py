import librosa
import numpy as np

def spectral_bandwidth_score(y, sr):

    """
    Measure spectral width/fullness.

    Big chorus/drop sections usually:
    - wider frequency spread
    - fuller instrumentation

    Returns:
        float
    """

    bandwidth = (
        librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr
        )[0]
    )

    return float(
        np.mean(bandwidth)
    )