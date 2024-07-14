import os
import json
import soundfile as sf
import numpy as np

from tqdm import tqdm
import time
import librosa
import sys


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



def yt2json(path_audio, path_json, output_path):

    # load
    wav, sr = sf.read(path_audio, always_2d=True)
    duration = len(wav) / sr
    with open(path_json ,'r') as f:
        json_str = f.read()
    yt_json = json.loads(json_str)

    mg_json = {"key": "", "artist": "", "sample_rate": sr, 
               "file_extension": "wav", "description": "", 
               "keywords": "", "duration": duration, "bpm": "", 
               "genre": "", "title": "", "name": "", "instrument": "Mix", "moods": []}
    
    mg_json["artist"] = yt_json["uploader"]
    mg_json["description"] = yt_json["title"]
    mg_json["keywords"] = ", ".join(yt_json["tags"])
    mg_json["name"] = yt_json["id"]
    mg_json["path"] = str(path_audio)
    
    with open(output_path, 'w') as js_file:
        json.dump(mg_json, js_file)
    

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

    # run
    err_files = [] 
    for i in range(st, ed): 
        print("==={}/{}======[{} - {}]========".format(
            i, num_files, st, ed))

        # path
        path_audio = audio_paths[i]
        dn = os.path.dirname(path_audio)
        json_dn = '/'.join(dn.split('/')[:-1]).replace('clip', 'full')
        path_json = os.path.join(json_dn, 'crawl_info.json') # replace the name of crawled json for each yt song here
        print(path_audio)
        print(path_json)
        output_path = path_audio.replace('no_vocal.wav', 'tags.json')

        # export abs midi
        try:
            yt2json(path_audio, path_json, output_path)
           
        except:
            print('[x] error')
            err_files.append(path_audio)
            sys.exit(1)
            continue
    print('\n\n\n==================')
    print('Error Files:')
    for idx, err_f in enumerate(err_files):
        print(idx, '-', err_f)