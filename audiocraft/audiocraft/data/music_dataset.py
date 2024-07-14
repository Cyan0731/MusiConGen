# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Dataset of music tracks with rich metadata.
"""
from dataclasses import dataclass, field, fields, replace
import gzip
import json
import logging
from pathlib import Path
import random
import typing as tp
import pretty_midi
import numpy as np

import torch
import torch.nn.functional as F
from .btc_chords import Chords

from .info_audio_dataset import (
    InfoAudioDataset,
    AudioInfo,
    get_keyword_list,
    get_keyword,
    get_string
)
from ..modules.conditioners import (
    ConditioningAttributes,
    JointEmbedCondition,
    WavCondition,
    ChordCondition,
    BeatCondition
)
from ..utils.utils import warn_once


logger = logging.getLogger(__name__)

CHORDS = Chords()


@dataclass
class MusicInfo(AudioInfo):
    """Segment info augmented with music metadata.
    """
    # music-specific metadata
    title: tp.Optional[str] = None
    artist: tp.Optional[str] = None  # anonymized artist id, used to ensure no overlap between splits
    key: tp.Optional[str] = None
    bpm: tp.Optional[float] = None
    genre: tp.Optional[str] = None
    moods: tp.Optional[list] = None
    keywords: tp.Optional[list] = None
    description: tp.Optional[str] = None
    name: tp.Optional[str] = None
    instrument: tp.Optional[str] = None
    chord: tp.Optional[ChordCondition] = None
    beat: tp.Optional[BeatCondition] = None
    # original wav accompanying the metadata
    self_wav: tp.Optional[WavCondition] = None
    # dict mapping attributes names to tuple of wav, text and metadata
    joint_embed: tp.Dict[str, JointEmbedCondition] = field(default_factory=dict)

    @property
    def has_music_meta(self) -> bool:
        return self.name is not None

    def to_condition_attributes(self) -> ConditioningAttributes:
        out = ConditioningAttributes()
        for _field in fields(self):
            key, value = _field.name, getattr(self, _field.name)
            if key == 'self_wav':
                out.wav[key] = value
            elif key == 'chord':
                out.chord[key] = value
            elif key == 'beat':
                out.beat[key] = value
            elif key == 'joint_embed':
                for embed_attribute, embed_cond in value.items():
                    out.joint_embed[embed_attribute] = embed_cond
            else:
                if isinstance(value, list):
                    value = ' '.join(value)
                out.text[key] = value
        return out

    @staticmethod
    def attribute_getter(attribute):
        if attribute == 'bpm':
            preprocess_func = get_bpm
        elif attribute == 'key':
            preprocess_func = get_musical_key
        elif attribute in ['moods', 'keywords']:
            preprocess_func = get_keyword_list
        elif attribute in ['genre', 'name', 'instrument']:
            preprocess_func = get_keyword
        elif attribute in ['title', 'artist', 'description']:
            preprocess_func = get_string
        else:
            preprocess_func = None
        return preprocess_func

    @classmethod
    def from_dict(cls, dictionary: dict, fields_required: bool = False):
        _dictionary: tp.Dict[str, tp.Any] = {}

        # allow a subset of attributes to not be loaded from the dictionary
        # these attributes may be populated later
        post_init_attributes = ['self_wav', 'chord', 'beat', 'joint_embed']
        optional_fields = ['keywords']

        for _field in fields(cls):
            if _field.name in post_init_attributes:
                continue
            elif _field.name not in dictionary:
                if fields_required and _field.name not in optional_fields:
                    raise KeyError(f"Unexpected missing key: {_field.name}")
            else:
                preprocess_func: tp.Optional[tp.Callable] = cls.attribute_getter(_field.name)
                value = dictionary[_field.name]
                if preprocess_func:
                    value = preprocess_func(value)
                _dictionary[_field.name] = value
        return cls(**_dictionary)


def augment_music_info_description(music_info: MusicInfo, merge_text_p: float = 0.,
                                   drop_desc_p: float = 0., drop_other_p: float = 0.) -> MusicInfo:
    """Augment MusicInfo description with additional metadata fields and potential dropout.
    Additional textual attributes are added given probability 'merge_text_conditions_p' and
    the original textual description is dropped from the augmented description given probability drop_desc_p.

    Args:
        music_info (MusicInfo): The music metadata to augment.
        merge_text_p (float): Probability of merging additional metadata to the description.
            If provided value is 0, then no merging is performed.
        drop_desc_p (float): Probability of dropping the original description on text merge.
            if provided value is 0, then no drop out is performed.
        drop_other_p (float): Probability of dropping the other fields used for text augmentation.
    Returns:
        MusicInfo: The MusicInfo with augmented textual description.
    """
    def is_valid_field(field_name: str, field_value: tp.Any) -> bool:
        valid_field_name = field_name in ['key', 'bpm', 'genre', 'moods', 'instrument', 'keywords']
        valid_field_value = field_value is not None and isinstance(field_value, (int, float, str, list))
        keep_field = random.uniform(0, 1) < drop_other_p
        return valid_field_name and valid_field_value and keep_field

    def process_value(v: tp.Any) -> str:
        if isinstance(v, (int, float, str)):
            return str(v)
        if isinstance(v, list):
            return ", ".join(v)
        else:
            raise ValueError(f"Unknown type for text value! ({type(v), v})")

    description = music_info.description

    metadata_text = ""
    # metadata_text = "rock style music, consistent rhythm, catchy song."
    if random.uniform(0, 1) < merge_text_p:
        meta_pairs = [f'{_field.name}: {process_value(getattr(music_info, _field.name))}'
                      for _field in fields(music_info) if is_valid_field(_field.name, getattr(music_info, _field.name))]
        random.shuffle(meta_pairs)
        metadata_text = ". ".join(meta_pairs)
        description = description if not random.uniform(0, 1) < drop_desc_p else None
        logger.debug(f"Applying text augmentation on MMI info. description: {description}, metadata: {metadata_text}")

    if description is None:
        description = metadata_text if len(metadata_text) > 1 else None
    else:
        description = ". ".join([description.rstrip('.'), metadata_text])
    description = description.strip() if description else None

    music_info = replace(music_info)
    music_info.description = description
    return music_info


class Paraphraser:
    def __init__(self, paraphrase_source: tp.Union[str, Path], paraphrase_p: float = 0.):
        self.paraphrase_p = paraphrase_p
        open_fn = gzip.open if str(paraphrase_source).lower().endswith('.gz') else open
        with open_fn(paraphrase_source, 'rb') as f:  # type: ignore
            self.paraphrase_source = json.loads(f.read())
        logger.info(f"loaded paraphrasing source from: {paraphrase_source}")

    def sample_paraphrase(self, audio_path: str, description: str):
        if random.random() >= self.paraphrase_p:
            return description
        info_path = Path(audio_path).with_suffix('.json')
        if info_path not in self.paraphrase_source:
            warn_once(logger, f"{info_path} not in paraphrase source!")
            return description
        new_desc = random.choice(self.paraphrase_source[info_path])
        logger.debug(f"{description} -> {new_desc}")
        return new_desc


class MusicDataset(InfoAudioDataset):
    """Music dataset is an AudioDataset with music-related metadata.

    Args:
        info_fields_required (bool): Whether to enforce having required fields.
        merge_text_p (float): Probability of merging additional metadata to the description.
        drop_desc_p (float): Probability of dropping the original description on text merge.
        drop_other_p (float): Probability of dropping the other fields used for text augmentation.
        joint_embed_attributes (list[str]): A list of attributes for which joint embedding metadata is returned.
        paraphrase_source (str, optional): Path to the .json or .json.gz file containing the
            paraphrases for the description. The json should be a dict with keys are the
            original info path (e.g. track_path.json) and each value is a list of possible
            paraphrased.
        paraphrase_p (float): probability of taking a paraphrase.

    See `audiocraft.data.info_audio_dataset.InfoAudioDataset` for full initialization arguments.
    """
    def __init__(self, *args, info_fields_required: bool = True,
                 merge_text_p: float = 0., drop_desc_p: float = 0., drop_other_p: float = 0.,
                 joint_embed_attributes: tp.List[str] = [],
                 paraphrase_source: tp.Optional[str] = None, paraphrase_p: float = 0,
                 **kwargs):
        kwargs['return_info'] = True  # We require the info for each song of the dataset.
        super().__init__(*args, **kwargs)
        self.info_fields_required = info_fields_required
        self.merge_text_p = merge_text_p
        self.drop_desc_p = drop_desc_p
        self.drop_other_p = drop_other_p
        self.joint_embed_attributes = joint_embed_attributes
        self.paraphraser = None
        self.downsample_rate = 640
        self.sr = 32000
        if paraphrase_source is not None:
            self.paraphraser = Paraphraser(paraphrase_source, paraphrase_p)

    def __getitem__(self, index):
        wav, info = super().__getitem__(index) # wav_seg and seg_info
        info_data = info.to_dict()

        # unpack info
        target_sr = self.sr
        n_frames_wave = info.n_frames
        n_frames_feat = int(info.n_frames // self.downsample_rate)

        music_info_path = str(info.meta.path).replace('no_vocal.wav', 'tags.json')
        chord_path = str(info.meta.path).replace('no_vocal.wav', 'chord.lab')
        beats_path = str(info.meta.path).replace('no_vocal.wav', 'beats.npy')

        if all([
            not Path(music_info_path).exists(),
            not Path(beats_path).exists(),
            not Path(chord_path).exists(),
        ]):
            raise FileNotFoundError

        ### music info        
        with open(music_info_path, 'r') as json_file:
            music_data = json.load(json_file)
            music_data.update(info_data)
            music_info = MusicInfo.from_dict(music_data, fields_required=self.info_fields_required)
        if self.paraphraser is not None:
                music_info.description = self.paraphraser.sample(music_info.meta.path, music_info.description)
        if self.merge_text_p:
            music_info = augment_music_info_description(
                music_info, self.merge_text_p, self.drop_desc_p, self.drop_other_p)
        

        ### load features to tensors ###
        feat_hz = target_sr/self.downsample_rate
        ## beat&bar: 2 x T
        feat_beats = np.zeros((2, n_frames_feat))
        
        beats_np = np.load(beats_path)
        beat_time = beats_np[:, 0]
        bar_time = beats_np[np.where(beats_np[:, 1] == 1)[0], 0]
        beat_frame = [
            int((t-info.seek_time)*feat_hz) for t in beat_time
                if (t >= info.seek_time and t < info.seek_time + self.segment_duration)]
        bar_frame =[
            int((t-info.seek_time)*feat_hz) for t in bar_time
                if (t >= info.seek_time and t < info.seek_time + self.segment_duration)]
        feat_beats[0, beat_frame] = 1
        feat_beats[1, bar_frame] = 1
        kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
        feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
        beat_events = feat_beats[0] + feat_beats[1]
        beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]

        music_info.beat = BeatCondition(beat=beat_events[None], length=torch.tensor([n_frames_feat]),
                                        bpm=[music_data["bpm"]], path=[music_info_path], seek_frame=[info.seek_time*target_sr//self.downsample_rate])

        ## chord: 12 x T
        feat_chord = np.zeros((12, n_frames_feat)) # root| ivs
        with open(chord_path, 'r') as f:
            for line in f.readlines():
                splits = line.split()
                if len(splits) == 3:
                    st_sec, ed_sec, ctag = splits
                    st_sec = float(st_sec) - info.seek_time
                    ed_sec = float(ed_sec) - info.seek_time
                    st_frame = int(st_sec*feat_hz)
                    ed_frame = int(ed_sec*feat_hz)
                    
                    # 12 chorma
                    mhot = CHORDS.chord(ctag)
                    final_vec = np.roll(mhot[2], mhot[0])

                    final_vec = final_vec[..., None]
                    feat_chord[:, st_frame:ed_frame] = final_vec
        feat_chord = torch.from_numpy(feat_chord)
        
        music_info.chord = ChordCondition(
                chord=feat_chord[None], length=torch.tensor([n_frames_feat]),
                bpm=[music_data["bpm"]], path=[chord_path], seek_frame=[info.seek_time*self.sr//self.downsample_rate])
            
        music_info.self_wav = WavCondition(
            wav=wav[None], length=torch.tensor([info.n_frames]),
            sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])

        for att in self.joint_embed_attributes:
            att_value = getattr(music_info, att)
            joint_embed_cond = JointEmbedCondition(
                wav[None], [att_value], torch.tensor([info.n_frames]),
                sample_rate=[info.sample_rate], path=[info.meta.path], seek_time=[info.seek_time])
            music_info.joint_embed[att] = joint_embed_cond

        return wav, music_info 


def get_musical_key(value: tp.Optional[str]) -> tp.Optional[str]:
    """Preprocess key keywords, discarding them if there are multiple key defined."""
    if value is None or (not isinstance(value, str)) or len(value) == 0 or value == 'None':
        return None
    elif ',' in value:
        # For now, we discard when multiple keys are defined separated with comas
        return None
    else:
        return value.strip().lower()


def get_bpm(value: tp.Optional[str]) -> tp.Optional[float]:
    """Preprocess to a float."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None
