# pipeline/features/structural.py

"""
Because in popular music:

chorus often more representative
intro often less important
bridge often unique but less representative

This is NOT final truth,
only:
soft prior knowledge.
"""

SECTION_WEIGHTS = {
    "chorus": 1.0,
    "verse": 0.7,
    "pre-chorus": 0.8,
    "bridge": 0.75,
    "intro": 0.4,
    "inst": 0.8,
    "outro": 0.4,
    "solo": 0.6
}


def structural_importance(candidate):
    """
    Estimate structural prior importance.

    Returns:
        float in [0, 1]
    """

    section = candidate["section"].lower()

    return SECTION_WEIGHTS.get(section, 0.5)