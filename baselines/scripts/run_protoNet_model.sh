#!/usr/bin/env bash
cd ..

python3 train_meta.py --config configs/train_protoNet.yaml --gpu 0 --seed 123 --tag ProtoNet --split_type NS 