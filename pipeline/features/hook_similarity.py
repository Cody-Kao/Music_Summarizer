import numpy as np

from pipeline.features.embedding import (
    cosine_sim
)


def hook_similarity(
    current_embedding,
    all_embeddings,
    current_index,
    min_distance=3
):

    """
    Detect repeated hook-like segments.

    Finds strongest similarity
    to non-neighbor repeated section.
    """

    best = 0.0

    for i, emb in enumerate(all_embeddings):

        if abs(i - current_index) < min_distance:
            continue

        sim = cosine_sim(
            current_embedding,
            emb
        )

        best = max(best, sim)

    return float(best)