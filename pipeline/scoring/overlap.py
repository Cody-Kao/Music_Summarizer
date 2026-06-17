# pipeline/scoring/overlap.py

def overlaps(a, b):

    return (
        max(
            a["start_time"],
            b["start_time"]
        )
        <
        min(
            a["end_time"],
            b["end_time"]
        )
    )