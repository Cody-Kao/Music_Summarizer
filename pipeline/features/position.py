def relative_position(
    candidate,
    song_duration
):

    center = (
        candidate["start_time"]
        + candidate["end_time"]
    ) / 2

    return center / song_duration