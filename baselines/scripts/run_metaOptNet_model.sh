#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_metaOptNet.yaml --gpu 0 --seed 123 --split_type NS
# python3 train_meta.py --config configs/configs_V2/train_metaOptNet_shapebd.yaml --gpu 0,1,2,3,4,5,6,7 --seed 124
# python3 train_meta.py --config configs/configs_V2/train_metaOptNet_shapebd.yaml --gpu 0,1,2,3,4,5,6,7 --seed 125