#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_maml.yaml --seed 123 --split_type NS --gpu 3