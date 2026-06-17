import collections
import collections.abc
import os
import numpy as np

from consts.consts import CANDIDATES_JSON_FOLDER_PATH, RESULT_FOLDER_PATH, SELECTED_CANDIDATES_JSON_FOLDER_PATH
from pipeline.features.demucs_utils import cleanup_temp_dir
from pipeline.features.salience_profile import get_dynamic_weights
from utils.load_files import get_file_infos
from utils.stitch_song import stitch_song_sections

# 1. Patch the removed aliases for collections
collections.MutableSequence = collections.abc.MutableSequence
collections.Sequence = collections.abc.Sequence
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable

# 2. Patch the removed aliases for legacy numpy (NumPy 2.0 Safe)
np.float = np.float64
np.int = np.int64
np.bool = np.bool_
np.object = np.object_
np.complex = np.complex128

# 3. Now import your pipeline
from pipeline.segmentation import beat_tracking
from pipeline.segmentation import candidate_generation
import json

from pipeline.features.extractor import (
    FeatureExtractor
)

from utils.convert_json import save_to_json

def pretty_print_features(features):
    return {
        k: round(v, 3)
        if isinstance(v, float)
        else v
        for k, v in features.items()
    }

def main():
    file_infos = get_file_infos()
    for filename, file_music_path, file_json_path in file_infos:
        try:
            # -----------------------------------
            # candidate generation
            # -----------------------------------
            beats = beat_tracking.extract_downbeats(file_music_path)
            bars = candidate_generation.build_bar_segments(beats)
            with open(file_json_path, 'r') as file:
                section = json.load(file)
            bars_with_section = candidate_generation.assign_bars_to_sections(bars, section)
            candidates = candidate_generation.generate_candidates(bars_with_section)

            # -----------------------------------
            # feature extraction
            # -----------------------------------
            feature_extractor = FeatureExtractor()

            candidates = (
                feature_extractor.enrich_candidates(
                    candidates,
                    file_music_path
                )
            )

            # -----------------------------------
            # scoring
            # -----------------------------------
            WEIGHTS = get_dynamic_weights(feature_extractor.profile)
            from pipeline.scoring.scoring import (
                attach_candidate_scores
            )

            attach_candidate_scores(candidates, WEIGHTS)

            # -----------------------------------
            # save to json
            # -----------------------------------
            candidate_json_path = os.path.join(CANDIDATES_JSON_FOLDER_PATH, f"{filename}_candidates.json")
            save_to_json(candidates, candidate_json_path)

            # -----------------------------------
            # PICK OPTIMAL COMBINATION OF CANDIDATES
            # -----------------------------------
            from pipeline.scoring.optimizer import (
                optimize_summary
            )

            selected_candidates = optimize_summary(
                candidates,
                max_duration=60
            )

            from pipeline.postprocess.boundary_refinement import (
                BoundaryRefiner
            )
            if feature_extractor.stem_paths:
                print("[Main] Separating vocals...")

                vocal_path = feature_extractor.stem_paths["vocals"]

                refiner = BoundaryRefiner(vocal_path)

                selected_candidates = (
                    refiner.refine_summary(
                        selected_candidates
                    )
                )

            for c in selected_candidates:
                print(
                    c["section"],
                    round(c["score"], 3),
                    round(c["duration"], 1),
                    c["start_time"],
                    c["end_time"]
                )
            print("="*20)
            for c in selected_candidates:
                c["start_time"] = c["refined_start_time"]
                c["end_time"] = c["refined_end_time"]
                print(
                    c["section"],
                    round(c["score"], 3),
                    round(c["duration"], 1),
                    round(c["refined_start_time"], 2),
                    round(c["refined_end_time"], 2)
                )    
                del c["refined_start_time"]
                del c["refined_end_time"]
            # -----------------------------------
            # Save selected candidates to json 
            # ----------------------------------- 
            candidate_json_path = os.path.join(SELECTED_CANDIDATES_JSON_FOLDER_PATH, f"{filename}_selected_candidates.json")
            save_to_json(selected_candidates, candidate_json_path)

            # -----------------------------------
            # SIMPLE STITCH INTO MP3
            # -----------------------------------
            stitch_song_sections(file_music_path, selected_candidates, os.path.join(RESULT_FOLDER_PATH, f"{filename}_summary.mp3"))
        finally:
            if feature_extractor.temp_dir:
                cleanup_temp_dir(feature_extractor.temp_dir)

if "__main__" == __name__:
    main()

"""
MIR/
├── main.py
├── pipeline/
│   ├── segmentation.py
│   ├── beat_tracking.py
│   ├── candidate_generation.py
│   ├── scoring.py
│   ├── selection.py
│   └── stitching.py
├── models/
├── data/
├── outputs/

------------------------------------

Audio
↓
SongFormer structure understanding
↓
Beat tracking
↓
Music-aware candidate generation
↓
stem calculate
↓
Candidate scoring
↓
Beam search optimization
↓
Boundary refinement
↓
Audio stitching
↓
Final summarized song

### stem calculate
Demucs stems
↓
vocal density
instrument density
↓
global song salience ratio
↓
detect:
    vocal-focused
    instrumental-focused
    mixed
↓
dynamic weight selection
↓
dynamic climax score
↓
candidate scoring
###

"""