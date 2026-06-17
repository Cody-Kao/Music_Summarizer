## Python Version 
The project is built with **Python 3.12.1**

## How to set up

**Set up virtual environment**
```
python -m venv .venv
```
**Then activate it**

Linux / Mac
```
source venv/bin/activate
```

Windows CMD
```
venv\Scripts\activate.bat
```

Windows Powershell
```
venv\Scripts\Activate.ps1
```

**Install packages**
```
pip install -r requirements.txt
```

**After installation**

change one line of codes in **.venv/Lib/site-packages/madmom/features/downbeats.py** at line 287
from `best = np.argmax(np.asarray(results)[:, 1])` to `best = np.argmax([r[1] for r in results])`

## Prepare music to process

1. A folder called **music** should contains all the music to process in the root directory.

2. A file called **input_list.scp** must be at the root directory, which contains the paths to music.

    e.g.
    ```txt
    music/Bach_Cello Suite No.1 in G major.mp3
    music/Your name.mp3
    music/Summer.mp3
    music/Libertango.mp3
    ```
    where **music** is the folder mentioned above 

3. A folder called **music_json** must present in root directory as well, which contains json files that are pre-processed by the SongFormer. The naming should be `{music_name}.json`. 

    **json files in music_json**

    ![example of json files in music_json](assets/image_1.png)

## How to run SongFormer

Result json files from SongFormer are necessary for this pipeline to work. Therefore, we've provided a Notebook on Kaggle to make everything convenient and doable.

[Notebook for SongFormer](https://www.kaggle.com/code/codykao/songformer)


## Run

After everything prepared, just run the system by
```py
python3 main.py
```

**About stitching segments: `dynamic_stitch_final.py`** 

## How to run `dynamic_stitch_final.py`
```python
python3 dynamic_stitch_final.py
```
---

## Getting output summary – Audio Stitching Module (`dynamic_stitch_final.py`)

This sub-module handles the intelligent post-processing, filtering, and seamless assembly of candidate music segments into a cohesive, production-ready audio summary. 

### Key Features

* **Dynamic Stitching Engine:** Dynamically selects transition modes (LPF burst, vocal anti-collision, smooth crossfade) and adapts overlap windows in real time based on MIR features.
* **Segment Deduplication:** Collaborates with micro-acoustic features (Chroma, Vocal Density, Spectral Flux) to prune redundant segments and prevent chorus fatigue.
* **Global Loudness Balancing:** Aligns junction decibels seamlessly using a two-step normalization pipeline and "Gain Ramps" to eliminate digital artifacts.
* **Peak Protection:** Built-in peak limiter that hard-limits maximum audio peaks to -1.0 dBFS to safely prevent digital clipping.
### Installation & Prerequisites

Ensure you have Python 3 and `ffmpeg` installed on your system. Install the required audio processing library:

```bash
pip install pydub
```

### How to Run
Execute the script directly to process the configured audio candidates and export the final optimized preview:

```Bash
python3 dynamic_stitch_final.py
```
