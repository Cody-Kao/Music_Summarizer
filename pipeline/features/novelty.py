# pipeline/features/novelty.py

"""
Avoid summaries becoming: average boring parts.

I want:
memorable moments
distinct timbre
section contrast
Simple Practical Approach

Novelty = distance from neighboring candidates.
Good summaries need: both representative AND distinctive moments.
"""

import numpy as np


def compute_song_centroid(
    embeddings
):
    """
    Average embedding of all candidates.
    """

    return np.mean(
        embeddings,
        axis=0
    )


def compute_novelty(
    candidate_embedding,
    song_centroid
):
    """
    Novelty relative to song identity.
    """

    sim = np.dot(
        candidate_embedding,
        song_centroid
    ) / (
        np.linalg.norm(candidate_embedding)
        * np.linalg.norm(song_centroid)
        + 1e-8
    )

    novelty = 1 - sim

    return float(novelty)