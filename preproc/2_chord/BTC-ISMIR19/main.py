'''
Chord Recognition with BTC
'''

import os
import mir_eval
import soundfile as sf
import pretty_midi as pm
from utils import logger
from btc_model import *
from utils.mir_eval_modules import audio_file_to_features, idx2chord, idx2voca_chord, get_audio_paths
import argparse
import warnings

import time
import datetime
import librosa

### ENV SETTING ###
warnings.filterwarnings('ignore')
# logger.logging_verbosity(1)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--voca', default=True, type=lambda x: (str(x).lower() == 'true'))
args = parser.parse_args()

config = HParams.load("run_config.yaml")

if args.voca is True:
    config.feature['large_voca'] = True
    config.model['num_chords'] = 170
    model_file = './test/btc_model_large_voca.pt'
    idx_to_chord = idx2voca_chord()
    # print("label type: large voca")
else:
    model_file = './test/btc_model.pt'
    idx_to_chord = idx2chord
    # print("label type: Major and minor")

model = BTC_model(config=config.model).to(device)
# Load model
if os.path.isfile(model_file):
    checkpoint = torch.load(model_file)
    mean = checkpoint['mean']
    std = checkpoint['std']
    model.load_state_dict(checkpoint['model'])
    print("restore model")

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
    base_audio = 'no_vocal'
    base_ext = 'wav'
    st, ed = 0, None


    audio_paths = traverse_dir(
            root_dir,
            str_include=base_audio,
            extension=base_ext,
            is_sort=True)
    num_files = len(audio_paths)
    
    if ed is None:
        ed = num_files
    print(' > num of files:', num_files)

    # Chord recognition and save lab file
    for i in range(st, ed): 
        print("==={}/{}======[{} - {}]========".format(
            i, num_files, st, ed))

        audio_path = audio_paths[i]
        save_path = os.path.join(
            os.path.dirname(audio_path), 'chord.lab')

        if os.path.exists(save_path):
            print('[o] existed')
            continue
        start_time_process = time.time()
        feature, feature_per_second, song_length_second = audio_file_to_features(
            audio_path, config)
        print(" > audio:", audio_path)

        # Majmin type chord recognition
        feature = feature.T
        feature = (feature - mean) / std
        time_unit = feature_per_second
        n_timestep = config.model['timestep']

        num_pad = n_timestep - (feature.shape[0] % n_timestep)
        feature = np.pad(feature, ((0, num_pad), (0, 0)), mode="constant", constant_values=0)
        num_instance = feature.shape[0] // n_timestep

        start_time = 0.0
        lines = []
        with torch.no_grad():
            model.eval()
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(device)
            for t in range(num_instance):
                self_attn_output, _ = model.self_attn_layers(
                    feature[:, n_timestep * t:n_timestep * (t + 1), :])
                prediction, _ = model.output_layer(self_attn_output)
                prediction = prediction.squeeze()
                for i in range(n_timestep):
                    if t == 0 and i == 0:
                        prev_chord = prediction[i].item()
                        continue
                    if prediction[i].item() != prev_chord:
                        lines.append(
                            '%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        start_time = time_unit * (n_timestep * t + i)
                        prev_chord = prediction[i].item()
                    if t == num_instance - 1 and i + num_pad == n_timestep:
                        if start_time != time_unit * (n_timestep * t + i):
                            lines.append('%.3f %.3f %s\n' % (start_time, time_unit * (n_timestep * t + i), idx_to_chord[prev_chord]))
                        break

        # lab file write
        print(' > save_path:', save_path)
        with open(save_path, 'w') as f:
            for line in lines:
                f.write(line)
        print(" > label:", save_path)

        # count time
        runtime = time.time() - start_time_process
        print(' [o] runtime:', str(datetime.timedelta(seconds=runtime)))
        print(' [o] RTF:', runtime/song_length_second)
