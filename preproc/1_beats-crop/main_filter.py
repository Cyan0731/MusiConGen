import os
import numpy as np
import math
import soundfile as sf
import json
import shutil
import uuid

import time
import datetime
PIVOT_RATIO = 0.8

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


def convert_to_decibel(arr, min_db=-120):
    ref = 1
    if arr!=0:
        return 20 * np.log10(abs(arr) / ref)
    else:
        return min_db


def compute_framewise_dbfs(
        signal, 
        win_len=1024, 
        hop_len=512):
    
    db_list = []
    for ed in range(win_len, signal.shape[0], hop_len):
        st = ed - win_len
        win_amplitude = np.mean(signal[st:ed, :])
        db_list.append(convert_to_decibel(win_amplitude))
    db_list = np.array(db_list)
    a = db_list < -80
    ratio = a.sum() / a.shape[0]
    return ratio


if __name__ == '__main__':

    start_time_all = time.time()
    root_dir = '../audiocraft/dataset/example/clip'
    files = traverse_dir(
        root_dir,
        str_include='no_vocal',
        extension='wav',
        is_sort=True)
    num_files = len(files)
    print(' > num of files:', num_files)

    #  save
    res = []
    ld_report = 'loudness_report_{}.txt'.format(str(uuid.uuid1()).split('-')[0])
    with open(ld_report, 'w') as f:
        for fidx in range(num_files):
            print('---({}/{})-------------'.format(fidx, num_files))
            file = files[fidx]
            signal, _ = sf.read(file, always_2d=True)
            ratio = compute_framewise_dbfs(signal) 
            print(file)
            print(ratio)
            res.append((file, ratio))

            f.write("{}-----:{}\n".format(file, ratio))

    
    with open(ld_report, 'r') as f:
        data = [line.strip().split('-----:') for line in f]

    # sort
    data = sorted(data, key=lambda x: float(x[1]))
    pivot = int(len(data) * PIVOT_RATIO)
    print('\n\n\n============================')
    print('pivot:', pivot)
    n_samples = len(data) - pivot
    not_ok_samples = data[-n_samples:]
    print('not ok samples:', n_samples)

    for i in range(n_samples):
        path_fn, ratio = not_ok_samples[i]
        print(path_fn, ratio)
        try:
            shutil.rmtree(os.path.dirname(path_fn))
        except:
            continue 

    # finish
    print('\n\n\n-------------------------------')
    print(' [o] Done')
    end_time_all = time.time()
    runtime = end_time_all - start_time_all
    print(f'Total time:', str(datetime.timedelta(seconds=runtime))+'\n')