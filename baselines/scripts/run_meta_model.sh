#!/usr/bin/env bash
cd ..
python3 train_meta.py --config configs/train_meta.yaml --gpu 4 --seed 123 --split_type NS
# python3 train_meta.py --config configs/train_meta.yaml --gpu 4 --seed 123 --split_type NS --pretrained_enc
