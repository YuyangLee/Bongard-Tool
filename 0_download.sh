#!bin/bash

nohup python 0_download.py --source pexels   > logs/pexels.out   &
nohup python 0_download.py --source unsplash > logs/unsplash.out &
nohup python 0_download.py --source flickr   > logs/flickr.out   &
