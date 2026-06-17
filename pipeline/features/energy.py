# pipeline/features/energy.py

"""
perceptual intensity.

Higher-energy regions often correspond to:

hooks
climax
emotionally salient moments


Feature	            Meaning
 RMS	             loudness
 Spectral Flux	     change intensity
 Zero Crossing Rate	 noisiness/percussiveness

Why These Features?
1. RMS(均方根: 描述音訊平均響度或功率的技術參數，用於測量聲音能量):
perceived loudness.
Often correlates with: emotional intensity and fullness

2. Spectral Flux(光譜通量/頻譜通量: 是一個在訊號處理、MIR和物理學中常用的概念，用來衡量訊號的功率譜（能量分佈）隨時間變化的快慢):
how rapidly spectrum changes.
High values often indicate: rhythmic activity, excitement, dense instrumentation

3. ZCR(過零率是指一個信號的符號變化的比率，例如信號從正變負數/或負變正，是分類敲擊聲的關鍵特徵):
noisiness / transient density, which is useful for distinguishing sparse vs active segments
"""

import librosa
import numpy as np


def compute_song_flux_stats(
    y,
    sr
):
    """
    Compute full-song spectral flux statistics.
    """

    flux = librosa.onset.onset_strength(
        y=y,
        sr=sr
    )

    return {
        "mean": np.mean(flux),
        "std": np.std(flux) + 1e-8
    }


def extract_energy_features(
    y,
    sr,
    flux_stats=None
):
    """
    Extract normalized energy features.
    """

    # -----------------------------------
    # RMS
    # -----------------------------------

    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))

    # -----------------------------------
    # Spectral Flux
    # -----------------------------------
    # compute candidate activity relative to song distribution
    flux = librosa.onset.onset_strength(
        y=y,
        sr=sr
    )

    flux_mean = np.mean(flux)

    if flux_stats:

        spectral_flux = (
            flux_mean
            - flux_stats["mean"]
        ) / flux_stats["std"]

    else:
        spectral_flux = flux_mean

    # -----------------------------------
    # ZCR
    # -----------------------------------

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(y)[0]
    )

    return {
        "rms": rms_mean,
        "spectral_flux": float(spectral_flux),
        "zcr": float(zcr)
    }