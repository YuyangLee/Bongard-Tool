#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_cnn.yaml --gpu 5 --seed 123 --split_type CGS