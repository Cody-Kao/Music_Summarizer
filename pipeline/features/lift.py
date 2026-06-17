def compute_lift(
    current_value,
    previous_value
):

    """
    Relative increase feature.

    Positive:
        stronger than previous

    Negative:
        weaker than previous
    """

    return (
        current_value
        - previous_value
    )