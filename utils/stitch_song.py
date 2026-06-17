from pydub import AudioSegment


def stitch_song_sections(
    audio_path: str,
    candidates: list,
    output_path: str,
    crossfade_ms: int = 200,
):
    """
    Stitch selected song sections into one audio summary.

    candidates format:
    [
        {
            "start_time": float,
            "end_time": float,
            ...
        }
    ]
    """

    song = AudioSegment.from_file(audio_path)

    combined = None

    for c in candidates:

        start_ms = int(c["start_time"] * 1000)
        end_ms = int(c["end_time"] * 1000)

        segment = song[start_ms:end_ms]

        if combined is None:
            combined = segment
        else:
            combined = combined.append(
                segment,
                crossfade=crossfade_ms
            )

    combined.export(output_path, format="mp3")

    print(f"Saved stitched summary to: {output_path}")