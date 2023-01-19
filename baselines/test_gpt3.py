# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import yaml
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import utils
import utils.few_shot as fs


def main(config):
    svname = args.name
    svname += args.split_type
    svname += '_' + config['model']

    save_path = os.path.join(args.save_dir, svname)
    utils.ensure_path(save_path, remove=False)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    #### Model ####
    

    #### Dataset ####
    n_way, n_shot = config['n_way'], config['n_shot']
    n_query = config['n_query']
    if config.get('n_train_way') is not None:
        n_train_way = config['n_train_way']
    else:
        n_train_way = n_way
    if config.get('ep_per_batch') is not None:
        ep_per_batch = config['ep_per_batch']
    else:
        ep_per_batch = 1

    random_state = np.random.RandomState(args.seed)
    print('seed：', args.seed)

    # split
    if args.split_type == 'NS':
        split = 'test'
    elif args.split_type == 'CGS':
        split = 'test_func'
    data_root = config['data_root']
    # test
    with open(f'{data_root}/{split}.json', 'r') as f:
        files = json.load(f)
    keys = list(files.keys())
    img_files = [files[key] for key in keys]
    concepts = keys
    for img_paths, concept in zip(img_files, concepts):
        pass

    ########
    ########


    writer.flush()
    print('finished training!')
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default='')
    parser.add_argument('--save_dir', default='./save')
    parser.add_argument('--split_type', default='NS', help='NS or CGS')
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--pretrained_enc', default=False, action='store_true')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)
    config['_gpu'] = args.gpu
    if len(args.gpu.split(',')) > 1:
        config['_parallel'] = True

    utils.set_gpu(args.gpu)
    main(config)
