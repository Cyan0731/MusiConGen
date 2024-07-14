apt-get update
apt-get install tmux vim git gcc -y
conda create -n beat python=3.9 -y
conda activate beat
# # pip install --upgrade --no-deps --force-reinstall 'git+https://github.com/CPJKU/madmom.git'
# pip install pyproject-toml
# # pip install git+https://github.com/CPJKU/madmom
pip install -e git+https://github.com/CPJKU/madmom#egg=madmom
pip install BeatNet
pip install torch==2.0.1
apt-get install portaudio19-dev -y
pip install pyaudio
conda install ffmpeg -y
pip install tqdm