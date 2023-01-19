#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_snail.yaml --gpu 3 --seed 123 --split_type CGS