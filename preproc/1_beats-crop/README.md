Two-stem demixing.

## Installation
```bash
source install.sh
```

## running
```bash
python main_beat_nn.py # much fater than main_beat.py (madmom)
python main_crop.py
python main_filter.py
```

## Monitoring
* for each full-length song
    * GPU: 
        * cpu: no
        * gpu: ~1G
    * Time: ~90 seconds

## BeatNet v.s. Madmom
* w5HP2Xcy_eQ: 4m30s 
    * beatnet (cpu): 8 sec
    * beatnet (gpu): 15 sec
    * madmom: 60 sec 
* 7_bo2zs50bg: 10m34s
    * beatnet (cpu): 13 sec
    * beatnet (gpu): 13 sec