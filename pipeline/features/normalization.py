import numpy as np

from pipeline.features.novelty import compute_novelty, compute_song_centroid

# function to normalize everything
def features_normalizer(candidates, candidate_embeddings):
    spectral_flux_normalization(candidates)
    vocal_density_normalization(candidates)
    rms_normalization(candidates)
    repetition_normalization(candidates)
    novelty_normalization(candidates, candidate_embeddings)
    normalize_feature(
        candidates,
        "drum_density"
    )
    normalize_feature(
        candidates,
        "bass_energy"
    )
    normalize_feature(
        candidates,
        "sustained_energy"
    )
    normalize_feature(
        candidates,
        "spectral_bandwidth"
    )


def normalize_feature(
    candidates,
    feature_name
):
    values = np.array([
        c["features"][feature_name]
        for c in candidates
    ])

    min_v = np.min(values)
    max_v = np.max(values)

    for candidate in candidates:

        raw = (
            candidate["features"]
            [feature_name]
        )

        norm = (
            raw - min_v
        ) / (
            max_v - min_v + 1e-8
        )

        candidate["features"][
            feature_name
        ] = float(norm)

# ==================================================
# Spectral Flux NORMALIZATION
# ==================================================
def spectral_flux_normalization(candidates):
    flux_values = np.array([
        c["features"]["spectral_flux"]
        for c in candidates
    ])

    min_f = np.min(flux_values)
    max_f = np.max(flux_values)

    for candidate in candidates:

        raw_f = (
            candidate["features"]
            ["spectral_flux"]
        )

        norm_f = (
            raw_f - min_f
        ) / (
            max_f - min_f + 1e-8
        )

        candidate["features"][
            "spectral_flux"
        ] = float(norm_f)

# ==================================================
# VOCAL DENSITY NORMALIZATION
# ==================================================
def vocal_density_normalization(candidates):
    print("[FeatureExtractor] Normalizing vocal density...")

    vocal_values = np.array([
        c["features"]["raw_vocal_density"]
        for c in candidates
    ])

    min_v = np.min(vocal_values)
    max_v = np.max(vocal_values)

    for candidate in candidates:

        raw_v = (
            candidate["features"]
            ["raw_vocal_density"]
        )

        normalized_v = (
            raw_v - min_v
        ) / (
            max_v - min_v + 1e-8
        )

        candidate["features"]["vocal_density"] = (
            float(normalized_v)
        )

        # optional:
        del candidate["features"][
            "raw_vocal_density"
        ]
    
    return candidates
    
# ==================================================
# RMS NORMALIZATION
# ==================================================
def rms_normalization(candidates):
    print("[FeatureExtractor] Normalizing RMS...")

    rms_values = np.array([
        c["features"]["rms"]
        for c in candidates
    ])

    min_r = np.min(rms_values)
    max_r = np.max(rms_values)

    for candidate in candidates:

        raw_rms = (
            candidate["features"]["rms"]
        )

        normalized_rms = (
            raw_rms - min_r
        ) / (
            max_r - min_r + 1e-8
        )

        candidate["features"]["rms"] = (
            float(normalized_rms)
        )

# =========================================
# REPETITION NORMALIZATION
# =========================================
def repetition_normalization(candidates):
    rep_values = np.array([
        c["features"]["raw_repetition"]
        for c in candidates
    ])

    min_r = np.min(rep_values)
    max_r = np.max(rep_values)

    for candidate in candidates:

        raw_r = (
            candidate["features"]
            ["raw_repetition"]
        )

        norm_r = (
            raw_r - min_r
        ) / (
            max_r - min_r + 1e-8
        )

        candidate["features"]["repetition"] = (
            float(norm_r)
        )

        del candidate["features"][
            "raw_repetition"
        ]

def novelty_normalization(candidates, candidate_embeddings):
    # ==================================================
    # GET GLOBAL NOVELTY for EACH CANDIDATE
    # ==================================================

    print("[FeatureExtractor] Computing novelty...")

    song_centroid = compute_song_centroid(
        candidate_embeddings
    )

    novelty_scores = []

    for candidate in candidates:

        novelty = compute_novelty(
            candidate["embedding"],
            song_centroid
        )

        novelty_scores.append(
            novelty
        )

    # ==================================================
    # NORMALIZE NOVELTY
    # ==================================================

    novelty_scores = np.array(
        novelty_scores
    )

    for i, candidate in enumerate(candidates):

        min_n = np.min(novelty_scores)
        max_n = np.max(novelty_scores)

        normalized_novelty = (
            novelty_scores[i] - min_n
        ) / (
            max_n - min_n + 1e-8
        )

        candidate["features"]["novelty"] = (
            float(normalized_novelty)
        )