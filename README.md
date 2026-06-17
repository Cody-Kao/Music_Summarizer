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