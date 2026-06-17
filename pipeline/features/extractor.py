# pipeline/features/extractor.py

"""
Feature Extraction Pipeline

full song
↓
compute song statistics
↓
Demucs separation once
↓
candidate loop
    ↓
    local features
    ↓
    embeddings
↓
global embedding statistics
↓
novelty computation
"""

import librosa
import numpy as np

from pipeline.features.sustained_energy import sustained_energy_score
from pipeline.features.position import relative_position
from pipeline.features.spectrum_bandwidth import spectral_bandwidth_score
from pipeline.features.structural import (
    structural_importance
)

from pipeline.features.instrument import (
    instrument_density
)

from pipeline.features.climax import (
    climax_score
)

from pipeline.features.salience_profile import (
    compute_vocal_ratio,
    detect_salience_profile
)

from pipeline.features.energy import (
    extract_energy_features,
    compute_song_flux_stats
)

from pipeline.features.demucs_utils import (
    separate_stems
)

from pipeline.features.vocal import (
    raw_vocal_density
)

from pipeline.features.embedding import (
    CLAPEmbeddingExtractor,
    cosine_sim
)

from pipeline.features.repetition import (
    repetition_score
)

from pipeline.features.harmony import (
    extract_chroma
)

from pipeline.features.lift import (
    compute_lift
)

from pipeline.features.hook_similarity import (
    hook_similarity
)


class FeatureExtractor:

    def __init__(self):

        # -----------------------------------
        # Load CLAP once
        # -----------------------------------

        self.embedding_extractor = (
            CLAPEmbeddingExtractor()
        )
        self.stem_paths = None 
        self.temp_dir = None 

    def enrich_candidates(
        self,
        candidates,
        audio_path
    ):

        # ==================================================
        # LOAD FULL SONG
        # ==================================================

        print("[FeatureExtractor] Loading full song...")

        full_y, full_sr = librosa.load(
            audio_path,
            sr=48000
        )
        full_song_duration = librosa.get_duration(y=full_y, sr=full_sr)
        # ==================================================
        # FULL SONG EMBEDDING
        # ==================================================

        print("[FeatureExtractor] Extracting full-song embedding...")

        full_song_embedding = (
            self.embedding_extractor
            .extract_embedding_from_audio(
                full_y,
                full_sr
            )
        )

        # ==================================================
        # FULL SONG ENERGY STATISTICS
        # ==================================================

        print("[FeatureExtractor] Computing song flux statistics...")

        flux_stats = compute_song_flux_stats(
            full_y,
            full_sr
        )

        # ==================================================
        # DEMUCS VOCAL SEPARATION
        # ==================================================

        print("[FeatureExtractor] Separating vocals with Demucs...")

        print(
            "[FeatureExtractor] Running Demucs..."
        )

        stem_paths, temp_dir = (separate_stems(audio_path))
        self.stem_paths = stem_paths
        self.temp_dir = temp_dir

        vocals_path = stem_paths["vocals"]

        drums_path = stem_paths["drums"]

        bass_path = stem_paths["bass"]

        other_path = stem_paths["other"]

        # ==================================================
        # CANDIDATE LOOP
        # ==================================================

        print("[FeatureExtractor] Processing candidates...")

        candidate_embeddings = []
        
        for idx, candidate in enumerate(candidates):
            # ----------------------------------------------
            # LOAD ORIGINAL CANDIDATE AUDIO
            # ----------------------------------------------
            print(f"\rTotal [FeatureExtractor] Candidate: {idx}/{len(candidates)}", end="", flush=True)
            y, sr = librosa.load(
                audio_path,
                sr=48000,
                offset=candidate["start_time"],
                duration=candidate["duration"]
            )

            # ----------------------------------------------
            # LOAD VOCAL, DRUM, BASS, Other SEGMENT
            # ----------------------------------------------

            vocals_y, _ = librosa.load(
                vocals_path,
                sr=48000,
                offset=candidate["start_time"],
                duration=candidate["duration"]
            )

            drums_y, _ = librosa.load(
                drums_path,
                sr=48000,
                offset=candidate["start_time"],
                duration=candidate["duration"]
            )

            bass_y, _ = librosa.load(
                bass_path,
                sr=48000,
                offset=candidate["start_time"],
                duration=candidate["duration"]
            )

            other_y, _ = librosa.load(
                other_path,
                sr=48000,
                offset=candidate["start_time"],
                duration=candidate["duration"]
            )

            features = {}

            # ==================================================
            # STRUCTURAL IMPORTANCE
            # ==================================================

            features["structural_importance"] = (
                structural_importance(candidate)
            )

            # ==================================================
            # ENERGY FEATURES
            # ==================================================

            features.update(
                extract_energy_features(
                    y,
                    sr,
                    flux_stats
                )
            )

            features.update(
                sustained_energy_score(
                    y,
                    sr
                )
            )

            features["spectral_bandwidth"] = (
                spectral_bandwidth_score(
                    y,
                    sr
                )
            )

            # ==================================================
            # VOCAL DENSITY
            # ==================================================

            features["raw_vocal_density"] = (
                raw_vocal_density(
                    vocals_y,
                    y
                )
            )

            # ==================================================
            # INSTRUMENT DENSITY
            # ==================================================
            drum_density, bass_energy, instrument_density_value = instrument_density(drums_y,bass_y,other_y)
            features["instrument_density"] = instrument_density_value

            # ==================================================
            # DRUM DENSITY & BASS ENERGY
            # ==================================================
            features["drum_density"] = drum_density
            features["bass_energy"] = bass_energy

            # ==================================================
            # Chroma Harmony
            # ==================================================

            features["chroma"] = (
                extract_chroma(y, sr)
            )

            # ==================================================
            # EMBEDDING
            # ==================================================

            embedding = (
                self.embedding_extractor
                .extract_embedding_from_audio(
                    y,
                    sr
                )
            )

            candidate["embedding"] = embedding

            candidate_embeddings.append(
                embedding
            )

            # ==================================================
            # REPRESENTATIVENESS
            # ==================================================

            features["representativeness"] = (
                cosine_sim(
                    embedding,
                    full_song_embedding
                )
            )

            # ==================================================
            # RAW Repetition
            # ==================================================

            features["raw_repetition"] = (
                repetition_score(y, sr)
            )

            features["relative_position"] = (
                relative_position(
                    candidate,
                    full_song_duration
                )
            )

            # ==================================================
            # SAVE FEATURES
            # ==================================================

            candidate["features"] = features
        
        # -----------------------------------
        # Global Features Normalization
        # -----------------------------------
        from pipeline.features.normalization import (
            features_normalizer, normalize_feature
        )
        features_normalizer(candidates, candidate_embeddings)

        # -----------------------------------
        # Energy Lift & Flux Lift
        # -----------------------------------
        for i in range(1, len(candidates)):

            current = candidates[i]
            previous = candidates[i - 1]

            current["features"]["energy_lift"] = (
                compute_lift(
                    current["features"]["rms"],
                    previous["features"]["rms"]
                )
            )

            current["features"]["flux_lift"] = (
                compute_lift(
                    current["features"]["spectral_flux"],
                    previous["features"]["spectral_flux"]
                )
            )
        candidates[0]["features"]["energy_lift"] = 0.0
        candidates[0]["features"]["flux_lift"] = 0.0
        normalize_feature(candidates, "energy_lift")
        normalize_feature(candidates, "flux_lift")

        # -----------------------------------
        # Calculate Hook Similarity
        # -----------------------------------
        all_embeddings = [
            c["embedding"]
            for c in candidates
        ]
        for i, candidate in enumerate(candidates):

            candidate["features"][
                "hook_similarity"
            ] = hook_similarity(
                candidate["embedding"],
                all_embeddings,
                i
            )

        # -----------------------------------
        # Calculate profile
        # -----------------------------------
        avg_vocal = np.mean([
                c["features"]["vocal_density"]
                for c in candidates
            ])

        avg_inst = np.mean([
            c["features"]["instrument_density"]
            for c in candidates
        ])

        vocal_ratio = compute_vocal_ratio(
            avg_vocal,
            avg_inst
        )

        profile = detect_salience_profile(
            vocal_ratio
        )
        self.profile = profile

        print("[FeatureExtractor] Profile:",self.profile)

        print("[FeatureExtractor] Vocal Ratio:",round(vocal_ratio, 3))

        # -----------------------------------
        # Add climax Score to features
        # -----------------------------------
        for candidate in candidates:
            candidate["features"]["climax"] = (
                climax_score(
                    candidate["features"]
                )
            )

        # -----------------------------------
        # Add Peak Score to features & Label Peak Sections
        # -----------------------------------
        from pipeline.features.peak_score import (
            peak_score, detect_peak
        )
        for candidate in candidates:
            candidate["features"]["peak_score"] = (
                peak_score(
                    candidate["features"]
                )
            )
        detect_peak(candidates)
        # ==================================================
        # DONE
        # ==================================================
        print("[FeatureExtractor] Done.")

        return candidates