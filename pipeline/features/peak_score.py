def peak_score(features):

    score = (

        0.35
        * features["sustained_energy"]

        + 0.20
        * features["energy_lift"]

        + 0.20
        * features["hook_similarity"]

        + 0.10
        * features["instrument_density"]

        + 0.10
        * features["spectral_bandwidth"]

        + 0.05
        * features["flux_lift"]
    )

    return float(score)

PEAK_PERCENTILE = 0.90
# -----------------------------------
# Detect peak candidates
# -----------------------------------
def detect_peak(candidates):
    peak_scores = sorted([
        c["features"]["peak_score"]
        for c in candidates
    ])

    threshold_index = int(
        len(peak_scores)
        * PEAK_PERCENTILE
    )

    threshold_index = min(
        threshold_index,
        len(peak_scores) - 1
    )

    peak_threshold = (
        peak_scores[threshold_index]
    )

    for candidate in candidates:

        candidate["is_peak"] = (

            candidate["features"]["peak_score"]
            >= peak_threshold
        )
