# pipeline/demucs_utils.py

import os
import sys
import shutil
import tempfile
import subprocess


def separate_stems(audio_path):

    """
    Run Demucs full stem separation.

    Returns:
        (
            stem_paths,
            output_dir
        )
    """

    output_dir = tempfile.mkdtemp()

    subprocess.run([
        sys.executable,
        "-m",
        "demucs",
        audio_path,
        "-o",
        output_dir
    ])

    song_name = os.path.splitext(
        os.path.basename(audio_path)
    )[0]

    base_dir = os.path.join(
        output_dir,
        "htdemucs",
        song_name
    )

    stem_paths = {

        "vocals": os.path.join(
            base_dir,
            "vocals.wav"
        ),

        "drums": os.path.join(
            base_dir,
            "drums.wav"
        ),

        "bass": os.path.join(
            base_dir,
            "bass.wav"
        ),

        "other": os.path.join(
            base_dir,
            "other.wav"
        )
    }

    return stem_paths, output_dir


def cleanup_temp_dir(
    output_dir
):

    """
    Remove temporary Demucs directory.
    """

    if os.path.exists(output_dir):

        shutil.rmtree(
            output_dir,
            ignore_errors=True
        )

        print(
            "[Demucs] Cleaned:",
            output_dir
        )