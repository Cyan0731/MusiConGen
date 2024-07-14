# MusiConGen

MusiConGen is a project for deep learning research on conditinoal music generation. MusiConGen is based on pretrained [Musicgen](https://github.com/facebookresearch/audiocraft) with additional controls: Rhythm and Chords. The project contains inference, training code and training data(youtube list).


## Installation
MusiConGen requires Python 3.9, PyTorch 2.0.0. To install MusiConGen, you can run the following:

```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
pip install 'torch>=2.0'
# Then proceed to one of the following
pip install -e .  # if you cloned the repo locally (mandatory if you want to train).
```

We also recommend having `ffmpeg` installed, either through your system or Anaconda:
```bash
sudo apt-get install ffmpeg
# Or if you are using Anaconda or Miniconda
conda install 'ffmpeg<5' -c  conda-forge
```

## Model

The model is based on the pretrained MusicGen-melody(1.5B). For infernece, GPU with VRAM greater than 12GB is recommended. For training, GPU with VRAM greater than 24GB is recommended.

## Inference

First, the model weight is at [link]().
Extract the model weight to directory `audiocraft/ckpt`.

One can simply run inference script with the command to generate music with chord and rhythm condition:
```shell
cd audiocraft
python generate_chord_beat.py
```

## Training 

### Training Data
The training data is provided as json format in 5_genre_songs_list.json. The listed suffixes are for youtube links.

### Data Preprocessing
Before training, one should put audio data in `audiocraft/dataset/$DIR_OF_YOUR_DATA$/full`.
And then run the preprocessing step by step:

```shell
cd preproc
```

### 1. demixing tracks
To remove the vocal stem from the track, we use [Demucs](https://github.com/facebookresearch/demucs).
In `main.py`, change `path_rootdir` to your directory and `ext_src` to the audio extention of your dataset (`'mp3'` or `'wav'`).

```shell
cd 0_demix
python main.py
```

### 2. beat/downbeat detection and cropping
To extract beat and down beat of songs, you can use [BeatNet](https://github.com/mjhydri/BeatNet) or [Madmom](https://github.com/CPJKU/madmom) as the beat extrctor.
For Beatnet user, change `path_rootdir` to your directory in `main_beat_nn.py`. For Madmom user, change `path_rootdir` to your directory in `main_beat.py`.

Then accroding to the extracted beat and downbeat, each song is cropped into clips in `main_crop.py`. `path_rootdir` should also be changed to your dataset directory.

The last stage is to filter out the clips with low volumn. `path_rootdir` should be changed to `clip` directory.

```shell
cd 1_beats-crop
python main_beat.py
python main_crop.py
python main_filter.py
```

### 3. chord extraction
To extract chord progression, we use [BTC-ISMIR2019](https://github.com/jayg996/BTC-ISMIR19).
The `root_dir` in `main.py` should be changed to your clips data directory.

```shell
cd 2_chord/BTC-ISMIR19
python main.py
```

### 4. tags/description labeling
For dataset crawled from website(e.g. youtube), the description of each song can be obtrained from crawled  informaiton `crawl_info.json`(you can change the file name in `3_1_ytjsons2tags/main.py`). We use the title of youtube song as description. The `root_dir` in `main.py` should be changed to your clips data directory.

```shell
cd 3_1_ytjsons2tags
python main.py
```

For dataset without information to describe, you can use [Essentia](https://github.com/MTG/essentia) to extract instrument and genre.
```shell
cd 3_tags/essentia
python main.py
```

### Training
The training weight of MusiConGen is at [link](). Please place it into the directory `MusiConGen/audiocraft/training_weights/xps/musicongen`.

Before training, you should set your username in environment variable
```shell
export env USER=$YOUR_USER_NAME
```

If using single gpu to finetune, you can use the following command:
```shell
dora run solver=musicgen/single_finetune \
    conditioner=chord2music_inattn.yaml \
    continue_from=//sig/musicongen \ 
    compression_model_checkpoint=//pretrained/facebook/encodec_32khz \
    model/lm/model_scale=medium dset=audio/example \
    transformer_lm.n_q=4 transformer_lm.card=2048
```
the `continue_from` argument can be also provided with your absolute path of your checkpoint. 

If you are using multiple(4) gpus to finetune, you can use the following command:
```shell
dora run -d solver=musicgen/multigpu_finetune \
    conditioner=chord2music_inattn.yaml \
    continue_from=//sig/musicongen \ 
    compression_model_checkpoint=//pretrained/facebook/encodec_32khz \
    model/lm/model_scale=medium dset=audio/example \
    transformer_lm.n_q=4 transformer_lm.card=2048
```

## export weight
use `export_weight.py` with your training signature `sig` to export your weight to `output_dir`.

## License
The license of code and model weights follows the LICENSE file, LICENSE of MusicGen in [LICENSE file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE) and [LICENSE_weights file](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights).