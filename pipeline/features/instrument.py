# pipeline/features/instrument.py

import numpy as np

def instrument_density(
    drums_y,
    bass_y,
    other_y
):
    """
    Estimate instrumental intensity.

    Drums:
        rhythmic energy

    Bass:
        low-end fullness

    Other:
        synths / guitars / harmonic richness
    """

    drum_energy = np.mean(
        drums_y ** 2
    )

    bass_energy = np.mean(
        bass_y ** 2
    )

    other_energy = np.mean(
        other_y ** 2
    )

    score = (
        0.4 * drum_energy
        + 0.2 * bass_energy
        + 0.4 * other_energy
    )

    return float(drum_energy), float(bass_energy), float(score)