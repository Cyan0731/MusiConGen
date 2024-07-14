apt-get update
conda create -n chord python=3.8 -y
conda activate chord
apt-get install vim tmux ffmpeg git rsync -y
cd /volume/ai-music-database/codes/data_processing/codes/2_chord/BTC-ISMIR19
pip install -r requirements.txt