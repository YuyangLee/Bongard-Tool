# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import argparse
import os
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from tensorboardX import SummaryWriter

import datasets
import models
import utils
import utils.few_shot as fs
from datasets.dataset import ToolDataModule


def main(config):
    svname = args.name
    svname += args.split_type
    svname += '_' + config['model']
    if args.pretrained_enc:
        svname += '_IN'
        config['model_args']['encoder_args']['pretrained'] = True
    else:
        svname += '_SC'
    if args.tag is not None:
        svname += '_' + args.tag

    save_path = os.path.join(args.save_dir, svname)
    utils.ensure_path(save_path, remove=False)
    utils.set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    yaml.dump(config, open(os.path.join(save_path, 'config.yaml'), 'w'))

    logger = utils.Logger(file_name=os.path.join(save_path, "log_sdout.txt"), file_mode="a+", should_flush=True)

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

    # dataset
    data_module = ToolDataModule(ep_per_batch, 8, config['data_root'], args.split_type, config['use_aug'])
    # train
    train_loader = data_module.train_dataloader()
    if config.get('visualize_datasets'):
        utils.visualize_dataset(train_loader, 'train_dataset', writer)
    # val
    val_loader = data_module.val_dataloader()
    if config.get('visualize_datasets'):
        utils.visualize_dataset(val_loader, 'val_dataset', writer)

    ########

    #### Model and optimizer ####

    if config.get('load'):
        print('loading pretrained model: ', config['load'])
        model = models.load(torch.load(config['load']))
    else:
        model = models.make(config['model'], **config['model_args'])

        if config.get('load_encoder'):
            print('loading pretrained encoder: ', config['load_encoder'])
            encoder = models.load(torch.load(config['load_encoder'])).encoder
            model.encoder.load_state_dict(encoder.state_dict())

        if config.get('load_prog_synthesis'):
            print('loading pretrained program synthesis model: ', config['load_prog_synthesis'])
            prog_synthesis = models.load(torch.load(config['load_prog_synthesis']))
            model.prog_synthesis.load_state_dict(prog_synthesis.state_dict())

    if config.get('_parallel'):
        model = nn.DataParallel(model)

    utils.log('num params: {}'.format(utils.compute_n_params(model)))

    optimizer, lr_scheduler = utils.make_optimizer(
        model.parameters(),
        config['optimizer'], **config['optimizer_args'])

    ########

    max_epoch = config['max_epoch']
    save_epoch = config.get('save_epoch')
    max_val_acc = 0.
    timer_used = utils.Timer()
    timer_epoch = utils.Timer()

    aves_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']

    trlog = dict()
    for k in aves_keys:
        trlog[k] = []


    if config.get('train_from_ckpt'):
        print('loading pretrained model: ', config['train_from_ckpt'])
        saved_obj = torch.load(config['train_from_ckpt'])
        model_sd = saved_obj['model_sd']
        model.load_state_dict(model_sd)
        training = saved_obj['training']
        epoch = training['epoch']
        optimizer_sd = training['optimizer_sd']
        optimizer.load_state_dict(optimizer_sd)
    else:
        epoch = 1


    while (epoch < max_epoch + 1):
        timer_epoch.s()
        aves = {k: utils.Averager() for k in aves_keys}

        # train
        model.train()
        if config.get('freeze_bn'):
            utils.freeze_bn(model)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        for data in tqdm(train_loader):

            x_shot, x_query = data['shot'].cuda(), data['query'].cuda()
            B = x_shot.size(0)
            label_query = fs.make_nk_label(
                n_train_way, n_query,
                ep_per_batch=B).cuda()

            if config['model'] == 'snail':  # only use one selected label_query
                query_dix = random_state.randint(n_train_way * n_query)
                # Fix BUG: the last iteration may not have ep_per_batch number of data
                # label_query = label_query.view(ep_per_batch, -1)[:, query_dix]
                label_query = label_query.view(B, -1)[:, query_dix]
                x_query = x_query[:, query_dix: query_dix + 1]

            if config['model'] == 'maml':  # need grad in maml
                model.zero_grad()

            logits = model(x_shot, x_query).view(-1, n_train_way)
            loss = F.cross_entropy(logits, label_query)
            acc = utils.compute_acc(logits, label_query)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            aves['train_loss'].add(loss.item())
            aves['train_acc'].add(acc)

            logits = None
            loss = None

        # eval
        model.eval()
        for data in tqdm(val_loader):
            x_shot, x_query = data['shot'].cuda(), data['query'].cuda()
            # shot: [B, 2, 6, 3, 128, 128], query: [B, 2, 1, 3, 128, 128]
            
            B = x_shot.size(0)
            
            label_query = fs.make_nk_label(
                n_train_way, n_query,
                ep_per_batch=B).cuda()

            if config['model'] == 'snail':  # only use one randomly selected label_query
                query_dix = random_state.randint(n_train_way)
                # Fix BUG: the last iteration may not have ep_per_batch number of data
                # label_query = label_query.view(ep_per_batch, -1)[:, query_dix]
                label_query = label_query.view(B, -1)[:, query_dix]
                x_query = x_query[:, query_dix: query_dix + 1]

            if config['model'] == 'maml':  # need grad in maml
                model.zero_grad()
                logits = model(x_shot, x_query, eval=True).view(-1, n_way)
                loss = F.cross_entropy(logits, label_query)
                acc = utils.compute_acc(logits, label_query)
            else:
                with torch.no_grad():
                    logits = model(x_shot, x_query, eval=True).view(-1, n_way)
                    loss = F.cross_entropy(logits, label_query)
                    acc = utils.compute_acc(logits, label_query)

            aves['val_loss'].add(loss.item())
            aves['val_acc'].add(acc)

        # post
        if lr_scheduler is not None:
            lr_scheduler.step()

        for k, v in aves.items():
            aves[k] = v.item()
            trlog[k].append(aves[k])

        t_epoch = utils.time_str(timer_epoch.t())
        t_used = utils.time_str(timer_used.t())
        t_estimate = utils.time_str(timer_used.t() / epoch * max_epoch)
        log_str = 'epoch {}, [train] loss {:.4f} | acc {:.4f}; [val] loss {:.4f} | acc {:.4f}'.format(
            epoch, aves['train_loss'], aves['train_acc'], aves['val_loss'], aves['val_acc'])
        log_str += ', {} {}/{}'.format(t_epoch, t_used, t_estimate)
        utils.log(log_str)

        writer.add_scalar('train_loss', aves['train_loss'], epoch)
        writer.add_scalar('train_acc', aves['train_acc'], epoch)
        writer.add_scalar('val_loss', aves['val_loss'], epoch)
        writer.add_scalar('val_acc', aves['val_acc'], epoch)

        if config.get('_parallel'):
            model_ = model.module
        else:
            model_ = model

        training = {
            'epoch': epoch,
            'optimizer': config['optimizer'],
            'optimizer_args': config['optimizer_args'],
            'optimizer_sd': optimizer.state_dict(),
        }
        save_obj = {
            'file': __file__,
            'config': config,

            'model': config['model'],
            'model_args': config['model_args'],
            'model_sd': model_.state_dict(),

            'training': training,
        }
        torch.save(save_obj, os.path.join(save_path, 'epoch-last.pth'))
        torch.save(trlog, os.path.join(save_path, 'trlog.pth'))

        if (save_epoch is not None) and epoch % save_epoch == 0:
            torch.save(save_obj,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if aves['val_acc'] > max_val_acc:
            max_val_acc = aves['val_acc']
            torch.save(save_obj, os.path.join(save_path, 'max-val_acc.pth'))

        writer.flush()

        epoch += 1

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
