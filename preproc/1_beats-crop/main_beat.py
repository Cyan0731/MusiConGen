import os
import uuid
import librosa
import soundfile as sf
import numpy as np

import time
import datetime
from tqdm import tqdm

import multiprocessing

from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor


def export_audio_with_click(proc_res, path_audio, path_output, sr=44100):
    # extract time
    times_beat = proc_res[np.where(proc_res[:, 1]!=1)][:, 0]
    times_downbeat = proc_res[np.where(proc_res[:, 1]==1)][:, 0]

    # load
    y, _ = librosa.core.load(path_audio, sr=sr) 

    # click audio
    y_beat = librosa.clicks(times=times_beat, sr=sr, click_freq=1200, click_duration=0.5) * 0.6
    y_downbeat = librosa.clicks(times=times_downbeat, sr=sr, click_freq=600, click_duration=0.5)

    # merge
    max_len = max(len(y), len(y_beat), len(y_downbeat))
    y_integrate = np.zeros(max_len)
    y_integrate[:len(y_beat)] += y_beat
    y_integrate[:len(y_downbeat)] += y_downbeat
    y_integrate[:len(y)] += y

    # librosa.output.write_wav(path_output, y_integrate, sr)
    sf.write(path_output, y_integrate, sr)


def estimate_beat(path_audio):
    # print('[*] estimating beats...')
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(path_audio)
    proc_res = proc(act) 
    return proc_res

def process(path_inpfile, path_outfile):
    pid = os.getpid()
    # start_time = time.time()
    # print(f'[PID: {pid}] > inp:', path_inpfile)
    # print(f'[PID: {pid}] > out:', path_outfile)

    if os.path.exists(path_outfile):
        print('[o] existed')
        return True
    
    # estimate beats
    beats = estimate_beat(path_inpfile)
    os.makedirs(os.path.dirname(path_outfile), exist_ok=True)
    np.save(path_outfile, beats)

    # export_audio_with_click(beats, path_inpfile, 'tmp.wav') # option
    # end
    # end_time = time.time()
    # runtime = end_time - start_time
    # print(f'[PID: {pid}] testing time:', str(datetime.timedelta(seconds=runtime))+'\n')
    return True


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


def gen_pairs(path_inpdir, audio_basename, ext):
    pairs = []
    filelist = traverse_dir(
            path_inpdir,
            extension=ext,
            str_include=audio_basename,
            is_sort=True)
    num_files = len(filelist)
    print(' > num of files (total):', num_files)

    for fidx in range(num_files): # p0
        path_inpfile = filelist[fidx]
        # path_outfile = os.path.join(
        #     os.path.dirname(path_inpfile), 'beats.npy')
        path_outfile = path_inpfile.replace('.wav', '.npy')

        if os.path.exists(path_outfile):
            print('[o] existed')
            continue

        pairs.append((path_inpfile, path_outfile))
    num_files = len(pairs)
    print(' > num of files (unprocessed):', num_files)
    return pairs, num_files

if __name__ == '__main__':
    path_rootdir = '../audiocraft/dataset/example/full'
    audio_basename = 'no_vocals'
    ext = 'wav'

    # list files
    pairs, num_files = gen_pairs(path_rootdir, audio_basename, ext)
    
    # count cpu
    cpu_count = 4
    print(' > cpu count:', cpu_count)

    start_time_all = time.time()
    with multiprocessing.Pool(processes=cpu_count) as pool, tqdm(total=num_files) as progress_bar:
        results = []
        for result in pool.starmap(process, pairs):
            results.append(result)
            progress_bar.update(1)

    # finish
    end_time_all = time.time()
    runtime = end_time_all - start_time_all
    print(f'testing time:', str(datetime.timedelta(seconds=runtime))+'\n')