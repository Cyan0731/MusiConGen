from audiocraft.utils import export
from audiocraft import train
import os
from pathlib import Path

sig = "your_training_signature"
output_dir = "./ckpt/output_weight_dir"


folder = f"./audiocraft_default/xps/{sig}"
export.export_lm(Path(folder) / 'checkpoint.th', os.path.join(output_dir, 'state_dict.bin'))
export.export_pretrained_compression_model('facebook/encodec_32khz', os.path.join(output_dir, 'compression_state_dict.bin'))