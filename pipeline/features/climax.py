# pipeline/features/climax.py
"""
    Adaptive climax score.

    Uses whichever salience
    channel dominates:
        vocals or instruments.
"""

def climax_score(
    features
):

    rms = features["rms"]

    repetition = (
        features["repetition"]
    )

    flux = (
        features["spectral_flux"]
    )

    drum = (
        features["drum_density"]
    )

    bass = (
        features["bass_energy"]
    )

    salience = max(

        features["vocal_density"],

        features["instrument_density"]
    )

    score = (

        rms*1.2
        * repetition
        * flux
        * salience
        * (0.5 + drum)
        * (0.5 + bass)

    ) ** (1 / 6)

    return float(score)