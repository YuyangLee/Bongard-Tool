# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the MIT License.
# To view a copy of this license, visit https://opensource.org/licenses/MIT

import os
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from typing import Any
import sys
from . import few_shot

import glob
import PIL.Image
import torchvision.utils as vutils

_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def set_gpu(gpu):
    print('set gpu:', gpu)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                       or input('{} exists, remove? ([y]/n): '.format(path)) != 'n'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def time_str(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    if t >= 60:
        return '{:.1f}m'.format(t / 60)
    return '{:.1f}s'.format(t)


def compute_logits(feat, proto, metric='dot', temp=1.0):
    assert feat.dim() == proto.dim()

    if feat.dim() == 2:
        if metric == 'dot':
            logits = torch.mm(feat, proto.t())
        elif metric == 'cos':
            logits = torch.mm(F.normalize(feat, dim=-1),
                              F.normalize(proto, dim=-1).t())
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(1) -
                       proto.unsqueeze(0)).pow(2).sum(dim=-1)

    elif feat.dim() == 3:
        if metric == 'dot':
            logits = torch.bmm(feat, proto.permute(0, 2, 1))
        elif metric == 'cos':
            logits = torch.bmm(F.normalize(feat, dim=-1),
                               F.normalize(proto, dim=-1).permute(0, 2, 1))
        elif metric == 'sqr':
            logits = -(feat.unsqueeze(2) -
                       proto.unsqueeze(1)).pow(2).sum(dim=-1)

    return logits * temp


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(params, name, lr, weight_decay=None, milestones=None):
    if weight_decay is None:
        weight_decay = 0.
    if name == 'sgd':
        optimizer = SGD(params, lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        optimizer = Adam(params, lr, weight_decay=weight_decay)
    if milestones:
        lr_scheduler = MultiStepLR(optimizer, milestones)
    else:
        lr_scheduler = None
    return optimizer, lr_scheduler


def visualize_dataset(dataloader, name, writer, n_samples=2):
    it = iter(dataloader)
    batch = next(it)
    shot = batch['shot'] # [B, 2, 6, 3, 128, 128]
    query = batch['query'] # [B, 2, 1, 3, 128, 128]
    imgs = torch.cat([shot, query], dim=2) # [B, 2, 7, 3, 128, 128]
    concepts = batch['concept'] # [B]
    B, _, _, C, H, W = imgs.shape
    print(imgs.shape)
    idx = torch.randperm(B)[:n_samples]
    imgs = imgs[idx]
    imgs = imgs * 0.5 + 0.5
    concepts = concepts[idx]
    print(imgs.shape)
    for i, concept in enumerate(concepts):
        img = vutils.make_grid(
                imgs[i].reshape(-1, C, H, W), normalize=False, nrow=7,
                padding=3, pad_value=0,
            )
        writer.add_images(f'visualize_{name}_task_{concept}', img, 0)

    writer.flush()


def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0:  # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------

def save_images2(images_A, images_B, size, image_path, is_permute):
    img = merge2(images_A, images_B, size, is_permute=is_permute)
    assert img.ndim == 3
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    fmt = 'RGB' if img.ndim == 3 else 'L'
    PIL.Image.fromarray(img, fmt).save(image_path)


def merge2(images_A, images_B, size=(3, 2), gap=20, is_permute=False):
    assert images_A.shape[0] == images_B.shape[0]
    h, w, c = images_A.shape[1], images_A.shape[2], images_A.shape[-1]

    test_idx_vec = np.array([0, 1])
    if is_permute:
        test_idx_vec = np.random.permutation([0, 1])

    test_images = [images_A[-1], images_B[-1]]

    img = np.zeros((h * size[0] + 2 * 2, w * size[1] * 2 + 3 * gap + 2 * 4 + w, c)) + 200
    for idx, image in enumerate(images_A[:-1]):
        i = idx % size[1]
        j = idx // size[1]
        pre_i = 0 if i == 0 else 2
        pre_j = 0 if j == 0 else 2
        img[h * j + pre_j * j:h * (j + 1) + pre_j * j, w * i + pre_i * i:w * (i + 1) + pre_i * i, :] = image

    for idx, image in enumerate(images_B[:-1]):
        i = idx % size[1] + size[1]
        j = idx // size[1]
        pre_i = 0 if i == 0 else 2
        pre_j = 0 if j == 0 else 2
        img[h * j + pre_j * j:h * (j + 1) + pre_j * j, gap + w * i + pre_i * i:gap + w * (i + 1) + pre_i * i, :] = image

    img[:h, 3 * gap + w * 4 + 2 * 4: 3 * gap + w * 5 + 2 * 4, :] = test_images[test_idx_vec[0]]
    img[h + 2: h * 2 + 2, 3 * gap + w * 4 + 2 * 4: 3 * gap + w * 5 + 2 * 4, :] = test_images[test_idx_vec[1]]

    return img


def vis_bongard(data_path, vis_path):
    filenames_A = sorted(glob.glob(os.path.join(data_path, '1', '*.png')))
    images_A_all = np.array([np.array(PIL.Image.open(fname)) for fname in filenames_A])
    print('images_A_all.shape: ', images_A_all.shape)
    filenames_B = sorted(glob.glob(os.path.join(data_path, '0', '*.png')))
    print('filenames_A[:1]: {}, filenames_B[:1]: {}'.format(filenames_A[:1], filenames_B[:1]))
    images_B_all = np.array([np.array(PIL.Image.open(fname)) for fname in filenames_B])

    save_images2(images_A_all, images_B_all, (3, 2), '{}/bongard_infer.png'.format(vis_path), is_permute=False)
