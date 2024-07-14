import os
import json

import sys
import librosa

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


if __name__ == '__main__':
    root_dir = '../audiocraft/dataset/example/clip'
    path_jsonl = '../audiocraft/egs/example/data.jsonl'

    filelist = traverse_dir(
        root_dir,
        extension='wav',
        str_include='no_vocal',
        is_sort=True)
    num_files = len(filelist)

    with open(path_jsonl, "w") as train_file:
    
        for fidx in range(num_files):
            print(f'==={fidx}/{num_files}================')
            path_wave = filelist[fidx]
            path_json = os.path.join(
                os.path.dirname(path_wave), 'tags.json')

            sr = librosa.get_samplerate(path_wave)
            
            print('path_wave:', path_wave)
            print('path_json:', path_json)

            with open(path_json, 'r') as f:
                data = json.load(f)
            assert sr == data['sample_rate']

            final = {
                'path': data['path'],
                'duration': data['duration'],
                "sample_rate": data['sample_rate'],
                "bpm": data['bpm'],
                "amplitude": None, 
                "weight": None, 
                "info_path": None
            }
            train_file.write(json.dumps(final) + '\n')
    print('\n\n\n==================')
    print('num files:', num_files)

   
