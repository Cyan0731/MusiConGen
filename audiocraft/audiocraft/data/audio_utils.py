# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Various utilities for audio convertion (pcm format, sample rate and channels),
and volume normalization."""
import sys
import typing as tp

import julius
import torch
import torchaudio
import numpy as np

from .chords import Chords
chords = Chords() # initiate object


def convert_audio_channels(wav: torch.Tensor, channels: int = 2) -> torch.Tensor:
    """Convert audio to the given number of channels.

    Args:
        wav (torch.Tensor): Audio wave of shape [B, C, T].
        channels (int): Expected number of channels as output.
    Returns:
        torch.Tensor: Downmixed or unchanged audio wave [B, C, T].
    """
    *shape, src_channels, length = wav.shape
    if src_channels == channels:
        pass
    elif channels == 1:
        # Case 1:
        # The caller asked 1-channel audio, and the stream has multiple
        # channels, downmix all channels.
        wav = wav.mean(dim=-2, keepdim=True)
    elif src_channels == 1:
        # Case 2:
        # The caller asked for multiple channels, but the input file has
        # a single channel, replicate the audio over all channels.
        wav = wav.expand(*shape, channels, length)
    elif src_channels >= channels:
        # Case 3:
        # The caller asked for multiple channels, and the input file has
        # more channels than requested. In that case return the first channels.
        wav = wav[..., :channels, :]
    else:
        # Case 4: What is a reasonable choice here?
        raise ValueError('The audio file has less channels than requested but is not mono.')
    return wav


def convert_audio(wav: torch.Tensor, from_rate: float,
                  to_rate: float, to_channels: int) -> torch.Tensor:
    """Convert audio to new sample rate and number of audio channels."""
    wav = julius.resample_frac(wav, int(from_rate), int(to_rate))
    wav = convert_audio_channels(wav, to_channels)
    return wav


def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                       loudness_compressor: bool = False, energy_floor: float = 2e-3):
    """Normalize an input signal to a user loudness in dB LKFS.
    Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

    Args:
        wav (torch.Tensor): Input multichannel audio data.
        sample_rate (int): Sample rate.
        loudness_headroom_db (float): Target loudness of the output in dB LUFS.
        loudness_compressor (bool): Uses tanh for soft clipping.
        energy_floor (float): anything below that RMS level will not be rescaled.
    Returns:
        torch.Tensor: Loudness normalized output data.
    """
    energy = wav.pow(2).mean().sqrt().item()
    if energy < energy_floor:
        return wav
    transform = torchaudio.transforms.Loudness(sample_rate)
    input_loudness_db = transform(wav).item()
    # calculate the gain needed to scale to the desired loudness level
    delta_loudness = -loudness_headroom_db - input_loudness_db
    gain = 10.0 ** (delta_loudness / 20.0)
    output = gain * wav
    if loudness_compressor:
        output = torch.tanh(output)
    assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
    return output


def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
    """Utility function to clip the audio with logging if specified."""
    max_scale = wav.abs().max()
    if log_clipping and max_scale > 1:
        clamp_prob = (wav.abs() > 1).float().mean().item()
        print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
              clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
    wav.clamp_(-1, 1)


def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                    strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                    rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                    loudness_compressor: bool = False, log_clipping: bool = False,
                    sample_rate: tp.Optional[int] = None,
                    stem_name: tp.Optional[str] = None) -> torch.Tensor:
    """Normalize the audio according to the prescribed strategy (see after).

    Args:
        wav (torch.Tensor): Audio data.
        normalize (bool): if `True` (default), normalizes according to the prescribed
            strategy (see after). If `False`, the strategy is only used in case clipping
            would happen.
        strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
            i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
            with extra headroom to avoid clipping. 'clip' just clips.
        peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
        rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
            than the `peak_clip` one to avoid further clipping.
        loudness_headroom_db (float): Target loudness for loudness normalization.
        loudness_compressor (bool): If True, uses tanh based soft clipping.
        log_clipping (bool): If True, basic logging on stderr when clipping still
            occurs despite strategy (only for 'rms').
        sample_rate (int): Sample rate for the audio data (required for loudness).
        stem_name (str, optional): Stem name for clipping logging.
    Returns:
        torch.Tensor: Normalized audio.
    """
    scale_peak = 10 ** (-peak_clip_headroom_db / 20)
    scale_rms = 10 ** (-rms_headroom_db / 20)
    if strategy == 'peak':
        rescaling = (scale_peak / wav.abs().max())
        if normalize or rescaling < 1:
            wav = wav * rescaling
    elif strategy == 'clip':
        wav = wav.clamp(-scale_peak, scale_peak)
    elif strategy == 'rms':
        mono = wav.mean(dim=0)
        rescaling = scale_rms / mono.pow(2).mean().sqrt()
        if normalize or rescaling < 1:
            wav = wav * rescaling
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    elif strategy == 'loudness':
        assert sample_rate is not None, "Loudness normalization requires sample rate."
        wav = normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
        _clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
    else:
        assert wav.abs().max() < 1
        assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
    return wav


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to float 32 bits PCM format.
    """
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / 2**15
    elif wav.dtype == torch.int32:
        return wav.float() / 2**31
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def i16_pcm(wav: torch.Tensor) -> torch.Tensor:
    """Convert audio to int 16 bits PCM format.

    ..Warning:: There exist many formula for doing this conversion. None are perfect
    due to the asymmetry of the int16 range. One either have possible clipping, DC offset,
    or inconsistencies with f32_pcm. If the given wav doesn't have enough headroom,
    it is possible that `i16_pcm(f32_pcm)) != Identity`.
    """
    if wav.dtype.is_floating_point:
        assert wav.abs().max() <= 1
        candidate = (wav * 2 ** 15).round()
        if candidate.max() >= 2 ** 15:  # clipping would occur
            candidate = (wav * (2 ** 15 - 1)).round()
        return candidate.short()
    else:
        assert wav.dtype == torch.int16
        return wav

def convert_txtchord2chroma_orig(text_chords, bpms, meters, gen_sec):
    chromas = []
    # total_len = int(gen_sec * 44100 / 512)
    total_len = int(gen_sec * 32000 / 640)
    for chord, bpm, meter in zip(text_chords, bpms, meters):
        phr_len = int(60. / bpm * (meter * 4) * 32000 / 640)
        # phr_len = int(60. / bpm * (meter * 4) * 44100 / 2048)
        chroma = torch.zeros([total_len, 12])
        count = 0
        offset = 0
        
        stext = chord.split(" ")
        timebin = phr_len // 4 # frames per bar
        while count < total_len:
            for tokens in stext:
                if count >= total_len: 
                    break
                stoken = tokens.split(',')
                for token in stoken:
                    off_timebin = timebin + offset
                    rounded_timebin = round(off_timebin)
                    offset = off_timebin - rounded_timebin
                    offset = offset/len(stoken)
                    add_step = rounded_timebin//len(stoken)
                    mhot = chords.chord(token)
                    rolled = np.roll(mhot[2], mhot[0])
                    for i in range(count, count + add_step):
                        if count >= total_len: 
                            break
                        chroma[i] = torch.Tensor(rolled)
                        count += 1
        chromas.append(chroma)
    chroma = torch.stack(chromas)
    return chroma

def convert_txtchord2chroma(chord, bpm, meter, gen_sec):
    total_len = int(gen_sec * 32000 / 640)

    phr_len = int(60. / bpm * (meter * 4) * 32000 / 640)
    # phr_len = int(60. / bpm * (meter * 4) * 44100 / 2048)
    chroma = torch.zeros([total_len, 12])
    count = 0
    offset = 0
    
    stext = chord.split(" ")
    timebin = phr_len // 4 # frames per bar
    while count < total_len:
        for tokens in stext:
            if count >= total_len: 
                break
            stoken = tokens.split(',')
            for token in stoken:
                off_timebin = timebin + offset
                rounded_timebin = round(off_timebin)
                offset = off_timebin - rounded_timebin
                offset = offset/len(stoken)
                add_step = rounded_timebin//len(stoken)
                mhot = chords.chord(token)
                rolled = np.roll(mhot[2], mhot[0])
                for i in range(count, count + add_step):
                    if count >= total_len: 
                        break
                    chroma[i] = torch.Tensor(rolled)
                    count += 1
    return chroma



def convert_txtchord2chroma_24(chord, bpm, meter, gen_sec):
    total_len = int(gen_sec * 32000 / 640)

    phr_len = int(60. / bpm * (meter * 4) * 32000 / 640)
    # phr_len = int(60. / bpm * (meter * 4) * 44100 / 2048)
    chroma = torch.zeros([total_len, 24])
    count = 0
    offset = 0
    
    stext = chord.split(" ")
    timebin = phr_len // 4 # frames per bar
    while count < total_len:
        for tokens in stext:
            if count >= total_len: 
                break
            stoken = tokens.split(',')
            for token in stoken:
                off_timebin = timebin + offset
                rounded_timebin = round(off_timebin)
                offset = off_timebin - rounded_timebin
                offset = offset/len(stoken)
                add_step = rounded_timebin//len(stoken)

                root, bass, ivs_vec, _ = chords.chord(token)
                root_vec = torch.zeros(12)
                root_vec[root] = 1
                final_vec = np.concatenate([root_vec, ivs_vec]) # [C]
                for i in range(count, count + add_step):
                    if count >= total_len: 
                        break
                    chroma[i] = torch.Tensor(final_vec)
                    count += 1
    return chroma

def get_chroma_chord_from_lab(chord_path, gen_sec):
    total_len = int(gen_sec * 32000 / 640)
    feat_hz = 32000/640
    intervals = []
    labels = []
    feat_chord = np.zeros((12, total_len)) # root| ivs
    with open(chord_path, 'r') as f:
        for line in f.readlines():
            splits = line.split()
            if len(splits) == 3:
                st_sec, ed_sec, ctag = splits
                st_sec = float(st_sec)
                ed_sec = float(ed_sec)

                st_frame = int(st_sec*feat_hz)
                ed_frame = int(ed_sec*feat_hz)

                mhot = chords.chord(ctag)
                final_vec = np.roll(mhot[2], mhot[0])

                final_vec = final_vec[..., None] # [C, T]
                feat_chord[:, st_frame:ed_frame] = final_vec
    feat_chord = torch.from_numpy(feat_chord)
    return feat_chord


def get_chroma_chord_from_text(text_chord, bpm, meter, gen_sec):
    total_len = int(gen_sec * 32000 / 640)

    phr_len = int(60. / bpm * (meter * 4) * 32000 / 640)
    chroma = np.zeros([12, total_len])
    count = 0
    offset = 0
    
    stext = chord.split(" ")
    timebin = phr_len // 4 # frames per bar
    while count < total_len:
        for tokens in stext:
            if count >= total_len: 
                break
            stoken = tokens.split(',')
            for token in stoken:
                off_timebin = timebin + offset
                rounded_timebin = round(off_timebin)
                offset = off_timebin - rounded_timebin
                offset = offset/len(stoken)
                add_step = rounded_timebin//len(stoken)
                mhot = chords.chord(token)
                final_vec = np.roll(mhot[2], mhot[0])
                final_vec = final_vec[..., None] # [C, T]

                for i in range(count, count + add_step):
                    if count >= total_len: 
                        break
                    chroma[:, i] = final_vec
                    count += 1
    feat_chord = torch.from_numpy(feat_chord)
    return feat_chord

def get_beat_from_npy(beat_path, gen_sec):
    total_len = int(gen_sec * 32000 / 640) 

    beats_np = np.load(beat_path, allow_pickle=True)
    feat_beats = np.zeros((2, total_len))
    meter = int(max(beats_np.T[1]))
    beat_time = beats_np[:, 0]
    bar_time = beats_np[np.where(beats_np[:, 1] == 1)[0], 0]

    beat_frame = [int((t)*feat_hz) for t in beat_time if (t >= 0 and t < duration)]
    bar_frame =[int((t)*feat_hz) for t in bar_time if (t >= 0 and t < duration)]

    feat_beats[0, beat_frame] = 1
    feat_beats[1, bar_frame] = 1
    kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
    feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
    beat_events = feat_beats[0] + feat_beats[1]
    beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]

    bpm = 60 // np.mean([j-i for i, j in zip(beat_time[:-1], beat_time[1:])])
    return beat_events, bpm, meter

def get_beat_from_bpm(bpm, meter, gen_sec):
    total_len = int(gen_sec * 32000 / 640)

    feat_beats = np.zeros((2, total_len))

    beat_time_gap = 60 / bpm
    beat_gap = 60 / bpm * feat_hz
    
    beat_time = np.arange(0, duration, beat_time_gap)
    beat_frame = np.round(np.arange(0, n_frames_feat, beat_gap)).astype(int)
    if beat_frame[-1] == n_frames_feat:
        beat_frame = beat_frame[:-1]
    bar_frame = beat_frame[::meter]
    
    feat_beats[0, beat_frame] = 1
    feat_beats[1, bar_frame] = 1
    kernel = np.array([0.05, 0.1, 0.3, 0.9, 0.3, 0.1, 0.05])
    feat_beats[0] = np.convolve(feat_beats[0] , kernel, 'same') # apply soft kernel
    beat_events = feat_beats[0] + feat_beats[1]
    beat_events = torch.tensor(beat_events).unsqueeze(0) # [T] -> [1, T]
    return beat_events, beat_time, meter