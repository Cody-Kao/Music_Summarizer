# pipeline/features/transition.py

"""
Measure:
how smoothly excerpts can connect.

Feature	                Why
 BPM difference	         rhythmic continuity
 RMS difference	         loudness continuity
 Embedding similarity	 timbral continuity
"""

# pipeline/scoring/transition.py

import numpy as np


def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
        + 1e-8
    )


def transition_score(
    candidate_a,
    candidate_b
):

    # -----------------------------------
    # Embedding continuity
    # -----------------------------------

    emb_sim = cosine_similarity(
        candidate_a["embedding"],
        candidate_b["embedding"]
    )

    # -----------------------------------
    # RMS continuity
    # -----------------------------------

    rms_diff = abs(
        candidate_a["features"]["rms"]
        - candidate_b["features"]["rms"]
    )

    rms_score = np.exp(
        -3 * rms_diff
    )

    """ # -----------------------------------
    # Structural continuity
    # -----------------------------------

    section_bonus = 0

    if (
        candidate_a["section"]
        == candidate_b["section"]
    ):
        section_bonus = 0.1 """
    
    # -----------------------------------
    # Chroma Harmony Continuity 
    # -----------------------------------
    
    harmonic_sim = cosine_similarity(
        candidate_a["features"]["chroma"],
        candidate_b["features"]["chroma"]
    )

    # -----------------------------------
    # Final
    # -----------------------------------

    score = (
        0.40 * emb_sim
        + 0.35 * rms_score
        + 0.25 * harmonic_sim
        #+ 0.1 * section_bonus
    )

    return float(score)