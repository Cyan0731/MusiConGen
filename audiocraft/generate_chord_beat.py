from audiocraft.data.audio import audio_write
import audiocraft.models
import numpy as np
import pandas as pd
import os
import torch

# set hparams
output_dir = 'example_1' ### change this output directory


duration = 30
num_samples = 5
bs = 1


# load your model
musicgen = audiocraft.models.MusicGen.get_pretrained('./ckpt/musicongen') ### change this path
musicgen.set_generation_params(duration=duration, extend_stride=duration//2, top_k = 250)


chords = ['C G A:min F',
          'A:min F C G',
          'C F G F',
          'C A:min F G',
          'D:min G C A:min',
          ]

descriptions = ["A laid-back blues shuffle with a relaxed tempo, warm guitar tones, and a comfortable groove, perfect for a slow dance or a night in. Instruments: electric guitar, bass, drums."] * num_samples

bpms = [120] * num_samples

meters = [4] * num_samples

wav = []
for i in range(num_samples//bs):
  print(f"starting {i} batch...")
  temp = musicgen.generate_with_chords_and_beats(descriptions[i*bs:(i+1)*bs], 
                                                  chords[i*bs:(i+1)*bs],
                                                  bpms[i*bs:(i+1)*bs], 
                                                  meters[i*bs:(i+1)*bs]
                                                  )
  wav.extend(temp.cpu())

# save and display generated audio
for idx, one_wav in enumerate(wav):
  
  sav_path = os.path.join('./output_samples', output_dir, chords[idx] + "|" + descriptions[idx]).replace(" ", "_")
  audio_write(sav_path, one_wav.cpu(), musicgen.sample_rate, strategy='loudness', loudness_compressor=True)