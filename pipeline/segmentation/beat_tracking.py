# pipeline/beat_tracking.py
from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor


def extract_downbeats(audio_path):
    """
    Returns:
        [
            {
                "time": float,
                "beat": int
            }
        ]
    """

    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[4], fps=100)
    act = RNNDownBeatProcessor()(audio_path)

    downbeats = proc(act)

    results = []

    for time, beat in downbeats:
        results.append({
            "time": float(time),
            "beat": int(beat)
        })

    return results