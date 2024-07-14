import os
from BeatNet.BeatNet import BeatNet

import time
import datetime
from tqdm import tqdm

import soundfile as sf
import librosa
import numpy as np


device = 'cuda' # 'cpu' or 'cuda', I found there is no difference

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


def estimate_beat_beatnet(path_audio):
    estimator = BeatNet(
        1, 
        mode='offline', 
        inference_model='DBN',
        plot=[], 
        thread=False, 
        device=device)
    
    beats = estimator.process(path_audio)
    return beats


def estimate_beat_madmom(path_audio):
    from madmom.features.downbeats import DBNDownBeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor
    # print('[*] estimating beats...')
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=100)
    act = RNNDownBeatProcessor()(path_audio)
    proc_res = proc(act) 
    return proc_res

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


if __name__ == '__main__':
    path_rootdir = '../audiocraft/dataset/example/full'
    audio_base = 'no_vocals'
    ext = 'wav'
    st, ed = 0, None


    filelist = traverse_dir(
        path_rootdir,
        extension=ext,
        str_include=audio_base,
        is_sort=True)
    num_files = len(filelist)
    print(' > num files:', num_files)
    if ed is None:
        ed = num_files

    # run
    start_time_all = time.time()

    for i in range(num_files-1,-1,-1):
        start_time_one = time.time()
        print("==={}/{}======[{} - {}]========".format(
            i, num_files, st, ed))
        path_audio = filelist[i]
        path_outfile = path_audio.replace('no_vocals.wav', 'beats.npy')
            

        print(' inp >', path_audio)
        print(' out >', path_outfile)
        if os.path.exists(path_outfile):
            print('[o] existed')
            continue
    
        beats = estimate_beat_beatnet(path_audio)

        # save
        np.save(path_outfile, beats)

        end_time_one = time.time()
        runtime = end_time_one - start_time_one
        print(f' > runtime:', str(datetime.timedelta(seconds=runtime))+'\n')

    end_time_all = time.time()
    runtime = end_time_all - start_time_all
    print(f'testing time:', str(datetime.timedelta(seconds=runtime))+'\n')

   