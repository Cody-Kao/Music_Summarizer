import os
from pathlib import Path

from consts.consts import INPUT_LIST_PATH, JSON_FOLDER_PATH

def _get_mp3_filepaths(folder):
    with open(folder, 'r') as f:
        filepaths = [line.strip() for line in f]
    return filepaths

def check_files_exist(filepaths):
    for filepath in filepaths:
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"{filepath} is missing or is not a valid file")

# List[tuple(filename, file_mp3_path, file_json_path)]
def get_file_infos(folder=INPUT_LIST_PATH):
    music_filepaths = _get_mp3_filepaths(folder)
    # check if all mp3 files exist in MUSIC_FOLDER_PATH
    check_files_exist(music_filepaths)
    # remove .mp3 ext name
    filenames = [Path(filepath.split("music")[1]).stem for filepath in music_filepaths]
    # check if all json files exist in JSON_FOLDER_PATH
    json_filepaths = [os.path.join(JSON_FOLDER_PATH, filename+".json") for filename in filenames]
    check_files_exist(json_filepaths)
    # zip filename and all paths together
    file_infos = [(filename, file_music_path, file_json_path) for filename, file_music_path, file_json_path in zip(filenames, music_filepaths, json_filepaths)]

    return file_infos

get_file_infos()