import numpy as np
import json
from pathlib import Path

def _convert_to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _convert_to_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_json(v) for v in obj]
    return obj

def save_to_json(obj, path, indent=2):
    # .parent gets the folder path (e.g., 'candidates_json') without the filename
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w") as file:
        json.dump(_convert_to_json(obj), file, indent=indent)