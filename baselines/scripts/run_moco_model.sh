#!/usr/bin/env bash
cd ..

python3 train_moco.py --config configs/configs_V2/train_moco_shapebd.yaml --gpu 0,1,2,3,4,5,6,7 --seed 123