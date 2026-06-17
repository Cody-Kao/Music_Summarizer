def compute_vocal_ratio(
    avg_vocal_density,
    avg_instrument_density
):

    """
    Compute vocal-vs-instrument ratio.

    Range:
        0 -> instrumental
        1 -> vocal
    """

    denom = (

        avg_vocal_density

        + avg_instrument_density

        + 1e-8
    )

    return (
        avg_vocal_density
        / denom
    )


def detect_salience_profile(
    vocal_ratio
):

    """
    Detect dominant salience type.
    """

    if vocal_ratio >= 0.65:

        return "vocal"

    elif vocal_ratio <= 0.35:

        return "instrumental"

    return "mixed"


def get_dynamic_weights(
    profile
):

    """
    Dynamic scoring weights.
    """

    # -----------------------------------
    # Vocal-focused
    # -----------------------------------

    if profile == "vocal":

        return {

            "climax": 0.30,

            "vocal_density": 0.18,

            "repetition": 0.18,

            "representativeness": 0.10,

            "instrument_density": 0.05,

            "rms": 0.07,

            "spectral_flux": 0.05,

            "novelty": 0.03,

            "structural_importance": 0.04
        }

    # -----------------------------------
    # Instrument-focused
    # -----------------------------------

    elif profile == "instrumental":

        return {

            "climax": 0.28,

            "instrument_density": 0.22,

            "spectral_flux": 0.15,

            "rms": 0.15,

            "repetition": 0.08,

            "vocal_density": 0.03,

            "representativeness": 0.05,

            "novelty": 0.02,

            "structural_importance": 0.02
        }

    # -----------------------------------
    # Mixed
    # -----------------------------------

    return {

        "climax": 0.30,

        "repetition": 0.15,

        "vocal_density": 0.12,

        "instrument_density": 0.12,

        "representativeness": 0.10,

        "rms": 0.08,

        "spectral_flux": 0.08,

        "novelty": 0.03,

        "structural_importance": 0.02
    }