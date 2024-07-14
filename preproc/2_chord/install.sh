apt-get update
conda create -n chord python=3.8 -y
conda activate chord
apt-get install vim tmux ffmpeg git rsync -y
cd BTC-ISMIR19
pip install -r requirements.txt