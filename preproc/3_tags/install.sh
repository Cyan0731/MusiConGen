conda create -n tags python=3.9 -y
conda activate tags
apt-get update
apt-get install tmux vim git gcc -y
apt-get install ffmpeg -y
pip install essentia-tensorflow
pip install tensorflow
pip install librosa