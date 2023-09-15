# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import mmcv
import os
import os.path as osp
import time
import torch
import torch.distributed as dist
import warnings
from mmcls import __version__
from medfmc.apis import init_random_seed, set_random_seed, train_model
from mmcls.datasets import build_dataset
from mmcls.models import build_classifier
from mmcls.utils import (auto_select_device, collect_env, get_root_logger,
                         setup_multi_processes)
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--config',default='D:/MedFM/configs/swin-b_vpt_try_update_exp1/', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--device', help='device used for training. (Deprecated)')
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--ipu-replicas',
        type=int,
        default=None,
        help='num of ipu replicas to use')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


args = parse_args()

tracking = ["endo"]
for track in tracking:
    list_track_dir = args.config + track + "/"
    list_dir = os.listdir(list_track_dir)
    for config_path_dir in list_dir:
        config_cls_dir = list_track_dir + config_path_dir + "/"
        list_config_dir = os.listdir(config_cls_dir)
        for config_dir in list_config_dir:
            print("staring", config_cls_dir + config_dir+ "/")
            config_stage_dir = config_cls_dir + config_dir + "/"
            list_config_dir_stage = os.listdir(config_stage_dir)
            for stage in list_config_dir_stage:
                cfg = Config.fromfile(config_stage_dir + stage)
                print(config_stage_dir + stage)
