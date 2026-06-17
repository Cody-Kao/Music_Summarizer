# pipeline/postprocess/boundary_refinement.py

import librosa
import numpy as np

from scipy.ndimage import gaussian_filter1d


class BoundaryRefiner:

    def __init__(
        self,
        vocal_audio_path,
        sr=22050,
        hop_length=512
    ):

        self.sr = sr
        self.hop_length = hop_length

        # -----------------------------------
        # Load vocal stem
        # -----------------------------------

        self.y, _ = librosa.load(
            vocal_audio_path,
            sr=sr
        )

        # -----------------------------------
        # RMS envelope
        # -----------------------------------

        rms = librosa.feature.rms(
            y=self.y,
            hop_length=hop_length
        )[0]

        # -----------------------------------
        # Smooth RMS
        # -----------------------------------

        self.rms = gaussian_filter1d(
            rms,
            sigma=3
        )

        self.times = librosa.frames_to_time(
            np.arange(len(self.rms)),
            sr=sr,
            hop_length=hop_length
        )

    # ==================================================
    # FIND LOW-ENERGY VALLEY
    # ==================================================

    def find_best_boundary(
        self,
        target_time,
        search_radius=0.35
    ):

        start_t = max(
            0,
            target_time - search_radius
        )

        end_t = (
            target_time + search_radius
        )

        start_idx = np.searchsorted(
            self.times,
            start_t
        )

        end_idx = np.searchsorted(
            self.times,
            end_t
        )

        if end_idx <= start_idx:
            return target_time

        local_rms = self.rms[
            start_idx:end_idx
        ]

        min_idx = np.argmin(local_rms)

        best_global_idx = (
            start_idx + min_idx
        )

        refined_time = self.times[
            best_global_idx
        ]

        return float(refined_time)

    # ==================================================
    # REFINE CANDIDATE
    # ==================================================

    def refine_candidate(
        self,
        candidate
    ):

        refined_start = (
            self.find_best_boundary(
                candidate["start_time"]
            )
        )

        refined_end = (
            self.find_best_boundary(
                candidate["end_time"]
            )
        )

        # -----------------------------------
        # Safety checks
        # -----------------------------------

        if refined_end <= refined_start:

            refined_start = (
                candidate["start_time"]
            )

            refined_end = (
                candidate["end_time"]
            )

        candidate[
            "refined_start_time"
        ] = refined_start

        candidate[
            "refined_end_time"
        ] = refined_end

        candidate[
            "refined_duration"
        ] = (
            refined_end
            - refined_start
        )

        return candidate

    # ==================================================
    # REFINE SUMMARY
    # ==================================================

    def refine_summary(
        self,
        selected_candidates
    ):

        refined = []

        for candidate in selected_candidates:

            refined.append(
                self.refine_candidate(
                    candidate
                )
            )

        return refined