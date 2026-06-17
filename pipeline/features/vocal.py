# pipeline/features/vocal.py

"""
Human listeners often remember: vocal-heavy segments.
Especially:
hooks
lyrical phrases
chorus vocals
Practical Feasible Approach

vocals ≈ harmonic content
drums ≈ percussive content

use Demucs, because:
mature
accurate
feasible
easy enough
dramatically better than HPSS

Not true vocal separation, but much better approximation.
"""

import numpy as np
import subprocess
import tempfile
import os
import sys


def separate_vocals(audio_path):

    """
    Run Demucs vocal separation once.
    Returns path to isolated vocals stem.
    """

    output_dir = tempfile.mkdtemp()


    subprocess.run([
        sys.executable,
        "-m",
        "demucs",
        audio_path,
        "-o",
        output_dir,
        "--two-stems=vocals"
    ])

    song_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    vocal_path = os.path.join(
        output_dir,
        "htdemucs",
        song_name,
        "vocals.wav"
    )

    return vocal_path, output_dir


def raw_vocal_density(
    vocal_y,
    original_y
):
    """
    Raw vocal energy ratio.

    Measures:
        vocal_energy / total_energy

    Returns:
        raw unnormalized value
    """

    vocal_energy = np.mean(
        vocal_y ** 2
    )

    total_energy = np.mean(
        original_y ** 2
    )

    ratio = vocal_energy / (
        total_energy + 1e-8
    )

    return float(ratio)