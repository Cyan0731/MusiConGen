import os

import numpy as np
import soundfile as sf
import librosa

import time
import datetime

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

CROP_LEN_SEC = 30

BAR_FIRST = 8

if __name__ == '__main__':
    start_time_all = time.time()

    path_dset = '../audiocraft/dataset/example'
    path_inpdir = os.path.join(path_dset, 'full')
    path_outdir = os.path.join(path_dset, 'clip')
    st, ed = 0, None

    filelist = traverse_dir(
        path_inpdir,
        extension='wav',
        str_include='no_vocal',
        is_pure=True,
        is_sort=True)
    num_files = len(filelist)
    ed = num_files if ed is None else ed
    print(' > num files:', num_files)

    for fidx in range(num_files-1, -1, -1):
        start_time_iter = time.time()
        print(f'==={fidx}/{num_files}====={st}-{ed}============')
        fn = filelist[fidx]
        dn = os.path.dirname(fn)
        path_audio = os.path.join(path_inpdir, fn)
        path_beats = os.path.join(path_inpdir, dn, 'beats.npy')

        print(fn)
        if not os.path.exists(path_audio):
            raise FileNotFoundError(path_beats)
        path_out_sndir = os.path.join(path_outdir, dn)

        if os.path.exists(path_out_sndir):
            print('[o] existed')
            continue

        # ==========
        try: 
            beats = np.load(path_beats)
            wav, sr = sf.read(path_audio, always_2d=True)
            duration =  len(wav) / sr
            print(' > wav:', wav.shape)
            print(' > sr: ', sr)
            print(' > ch: ', wav.shape[1])
            print(' > duration:', len(wav) / sr)

            bar_idx = np.where(beats[:, 1] == 1)[0]
            num_bars = len(bar_idx)
            print(' > number of bars:', num_bars)

            BAR_HOP = int(30 / (duration  / num_bars))
            print(' > bar hop:', BAR_HOP)

            bar_starts = [bar_idx[i] for i in range(3, len(bar_idx), BAR_HOP)]

            clip_starts = []
            for bs in bar_starts:
                item = (
                    beats[bs, 0], # seconds
                    bs # index
                )
                clip_starts.append(item)

            max_sample = wav.shape[0] - 10*sr
            CLIP_LEN_SAMPLE = CROP_LEN_SEC*sr

            # crop
            count_clips = 0
            for uid, (clip_st_sec, bidx) in enumerate(clip_starts):
                # boundaries
                clip_ed_sec = clip_st_sec + CROP_LEN_SEC
                clip_st_sample = int(clip_st_sec*sr)
                clip_ed_sample = clip_st_sample + CLIP_LEN_SAMPLE
                if clip_ed_sample > max_sample:
                    break
                
                # crop
                clip_wav = wav[clip_st_sample:clip_ed_sample]
                clip_beats = []
                
                for bi in range(bidx, len(beats)):
                    if beats[bi, 0] < clip_ed_sec:
                        clip_beats.append(
                            [beats[bi, 0]-clip_st_sec, beats[bi, 1]]
                        )
                
                # save
                path_out_audio_clip = os.path.join(
                    path_out_sndir, str(bidx),'no_vocal.wav')

                if os.path.exists(path_out_audio_clip):
                    print('[o] existed')
                    continue
                
                path_out_beats_clip = os.path.join(
                    path_out_sndir, str(bidx), 'beats.npy')
                os.makedirs(
                    os.path.dirname(path_out_audio_clip), exist_ok=True)
                sf.write(path_out_audio_clip, clip_wav, sr)
                np.save(path_out_beats_clip, clip_beats)

                count_clips += 1
            print(' > count:',  count_clips)
        except:
            print('[x] aborted')
            continue 

        # finish
        end_time_iter = time.time()
        runtime = end_time_iter - start_time_iter
        print(f'testing time:', str(datetime.timedelta(seconds=runtime))+'\n')
            

    # finish
    print('\n\n\n-------------------------------')
    print(' [o] Done')
    end_time_all = time.time()
    runtime = end_time_all - start_time_all
    print(f'Total time:', str(datetime.timedelta(seconds=runtime))+'\n')

