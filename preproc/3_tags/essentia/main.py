import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import subprocess as sp
import librosa

from metadata import genre_labels, mood_theme_classes, instrument_classes
import numpy as np

import sys
import time
import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb", "--output", "genre_discogs400-discogs-effnet-1.pb"])
# sp.call(["curl", "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb", "--output", "discogs-effnet-bs64-1.pb"])
# sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb", "--output", "mtg_jamendo_moodtheme-discogs-effnet-1.pb"])
# sp.call(["curl", "https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb", "--output", "mtg_jamendo_instrument-discogs-effnet-1.pb"])

import sys

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from essentia.standard import (
    MonoLoader,
    TensorflowPredictEffnetDiscogs,
    TensorflowPredict2D,
)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def filter_predictions(predictions, class_list, threshold=0.1):
    predictions_mean = np.mean(predictions, axis=0)
    sorted_indices = np.argsort(predictions_mean)[::-1]
    filtered_indices = [i for i in sorted_indices if predictions_mean[i] > threshold]
    filtered_labels = [class_list[i] for i in filtered_indices]
    filtered_values = [predictions_mean[i] for i in filtered_indices]
    return filtered_labels, filtered_values

def make_comma_separated_unique(tags):
    seen_tags = set()
    result = []
    for tag in ', '.join(tags).split(', '):
        if tag not in seen_tags:
            result.append(tag)
            seen_tags.add(tag)
    return ', '.join(result)

# embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
# genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
# mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
# instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")

def get_audio_features(audio_filename):
    audio = MonoLoader(filename=audio_filename, sampleRate=16000, resampleQuality=4)()
    embedding_model = TensorflowPredictEffnetDiscogs(graphFilename="discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
    embeddings = embedding_model(audio)

    result_dict = {}

    # Predicting genres
    genre_model = TensorflowPredict2D(graphFilename="genre_discogs400-discogs-effnet-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall:0")
    predictions = genre_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, genre_labels)
    filtered_labels = ', '.join(filtered_labels).replace("---", ", ").split(', ')
    result_dict['genres'] = make_comma_separated_unique(filtered_labels)

    # Predicting mood/theme
    mood_model = TensorflowPredict2D(graphFilename="mtg_jamendo_moodtheme-discogs-effnet-1.pb")
    predictions = mood_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, mood_theme_classes, threshold=0.05)
    result_dict['moods'] = make_comma_separated_unique(filtered_labels)

    # Predicting instruments
    instrument_model = TensorflowPredict2D(graphFilename="mtg_jamendo_instrument-discogs-effnet-1.pb")
    predictions = instrument_model(embeddings)
    filtered_labels, _ = filter_predictions(predictions, instrument_classes)
    result_dict['instruments'] = filtered_labels

    return result_dict


def test():
    filename = 'Mr_Blue_Sky_Pomplamoose.mp3'

    # extract features
    result = get_audio_features(str(filename))

    # load audio
    sr = librosa.get_samplerate(str(filename))
    y, sr_load = librosa.load(str(filename), sr=sr)
    length = librosa.get_duration(y=y, sr=sr)
    assert sr == sr_load

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(tempo)

    # get key
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = np.argmax(np.sum(chroma, axis=1))
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]

    # entry
    entry = {
        "key": f"{key}",
        "artist": "",
        "sample_rate": sr,
        "file_extension": "wav",
        "description": "",
        "keywords": "",
        "duration": length,
        "bpm": tempo,
        "genre": result.get('genres', ""),
        "title": "",
        "name": "",
        "instrument": result.get('instruments', ""),
        "moods": result.get('moods', []),
        "path": str(filename),
    }

    # save
    with open(str(filename).rsplit('.', 1)[0] + '.json', "w") as file:
        json.dump(entry, file)


def traverse_dir(
        root_dir,
        extension,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def process_one(filename):
    dn = os.path.dirname(str(filename))
    path_outfile = os.path.join(dn, 'tags.json')
    if os.path.exists(path_outfile):
        print('[o] exsited')
        return

    # extract features
    result = get_audio_features(str(filename))

    # load audio
    sr = librosa.get_samplerate(str(filename))
    y, sr_load = librosa.load(str(filename), sr=sr)
    assert sr==sr_load

    # tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = round(tempo)

    # get key
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    key = np.argmax(np.sum(chroma, axis=1))
    key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key]

    # get duration
    length = librosa.get_duration(y=y, sr=sr)

    genre = result.get('genres', "")
    instr = result.get('instruments', "")
    description = f"{genre} style music with instrument: {', '.join(instr)}"

    # entry
    entry = {
        "key": f"{key}",
        "artist": "",
        "sample_rate": sr,
        "file_extension": "wav",
        "description": description,
        "keywords": "",
        "duration": length,
        "bpm": tempo,
        "genre": genre,
        "title": "",
        "name": "",
        "instrument": instr,
        "moods": result.get('moods', []),
        "path": str(filename),
    }

    # save
    print('[o] save to', path_outfile)
    with open(path_outfile, "w") as file:
        json.dump(entry, file)

if __name__ == '__main__':
    root_dir = '../audiocraft/dataset/example/clip'

    base_audio = 'no_vocal'
    base_ext = 'wav'
    st, ed = 0, None

    audio_paths = traverse_dir(
            root_dir,
            str_include=base_audio,
            extension=base_ext,
            is_sort=True)
    num_files = len(audio_paths)
    print(' > num of files:', num_files)
    if ed is None:
        ed = num_files

    # Chord recognition and save lab file
    for i in range(st, ed): 
        print("==={}/{}======[{} - {}]========".format(
            i, num_files, st, ed))
        
        filename = audio_paths[i]
        print(filename)
        
        start_time = time.time()
        try:
            process_one(filename)
        except:
            print('[x] aborted')
            continue
        end_time = time.time()
        runtime = end_time - start_time
        print('testing time:', str(datetime.timedelta(seconds=runtime))+'\n')