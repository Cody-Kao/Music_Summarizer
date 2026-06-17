import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


AUDIO_PATH = "music/MAGO-GFRIEND.mp3"


# =====================================================
# LOAD AUDIO
# =====================================================

y, sr = librosa.load(
    AUDIO_PATH,
    sr=22050
)

duration = librosa.get_duration(
    y=y,
    sr=sr
)

print("Duration:", duration)


# =====================================================
# RMS ENERGY
# =====================================================

hop_length = 512

rms = librosa.feature.rms(
    y=y,
    hop_length=hop_length
)[0]

rms_times = librosa.times_like(
    rms,
    sr=sr,
    hop_length=hop_length
)


# =====================================================
# SPECTRAL FLUX
# =====================================================

onset_env = librosa.onset.onset_strength(
    y=y,
    sr=sr
)

flux_times = librosa.times_like(
    onset_env,
    sr=sr
)


# =====================================================
# CHROMA
# =====================================================

chroma = librosa.feature.chroma_cqt(
    y=y,
    sr=sr
)


# =====================================================
# SELF-SIMILARITY MATRIX
# =====================================================

chroma_norm = librosa.util.normalize(
    chroma,
    axis=0
)

ssm = np.dot(
    chroma_norm.T,
    chroma_norm
)


# =====================================================
# PLOT RMS + FLUX
# =====================================================

plt.figure(figsize=(16, 5))

plt.plot(
    rms_times,
    rms,
    label="RMS Energy"
)

plt.plot(
    flux_times,
    onset_env / np.max(onset_env),
    label="Spectral Flux"
)

plt.xlabel("Time (s)")
plt.title("Energy + Spectral Flux")
plt.legend()

plt.show()


# =====================================================
# PLOT CHROMA
# =====================================================

plt.figure(figsize=(16, 5))

librosa.display.specshow(
    chroma,
    x_axis="time",
    y_axis="chroma"
)

plt.colorbar()

plt.title("Chroma Features")

plt.show()


# =====================================================
# PLOT SELF-SIMILARITY MATRIX
# =====================================================

plt.figure(figsize=(10, 10))

plt.imshow(
    ssm,
    origin="lower",
    aspect="auto",
    cmap="magma"
)

plt.xlabel("Time Frame")
plt.ylabel("Time Frame")

plt.title(
    "Self-Similarity Matrix"
)

plt.colorbar()

plt.show()