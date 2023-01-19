#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_metaOptNet.yaml --gpu 0 --seed 123 --split_type NS