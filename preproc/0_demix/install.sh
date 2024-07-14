cd /volume/wayne-nas-ai-music/projects/mir-pipeline/demix/demucs
apt-get update
apt-get install tmux vim -y
conda env update -f environment-cuda.yml
conda activate demucs
pip install -e .
conda update ffmpeg
cd /volume/wayne-nas-ai-music/projects/mir-pipeline/demix