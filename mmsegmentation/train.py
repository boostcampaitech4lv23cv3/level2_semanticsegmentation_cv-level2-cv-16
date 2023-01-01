# 모듈 import
import platform
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.datasets import (build_dataloader, build_dataset)
from mmseg.utils import get_device
from multiprocessing import freeze_support

from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed
from mmseg.utils import collect_env, get_root_logger

import wandb
import wandb_config

from mmcv.runner.hooks import HOOKS, Hook


selfos = platform.system() 

model_dir = 'segformer'
model_name = 'segformer_mit-b5_640x640_160k_ade20k'
work_dir = f'./work_dirs/{model_name}'
data_root = '/opt/ml/input/data'

def train(k_fold):

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')

    #get k_fold
    cfg.data.train.img_dir = data_root + f'/images/train_{k_fold}'
    cfg.data.train.ann_dir = data_root + f'/annotations/train_{k_fold}'
    cfg.data.val.img_dir   = data_root + f'/images/val_{k_fold}'
    cfg.data.val.ann_dir   = data_root + f'/annotations/val_{k_fold}'
    
    cfg.data.workers_per_gpu = 4 #num_workers
    cfg.data.samples_per_gpu = 3

    cfg.seed = 42
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir+f'_{k_fold}'

    cfg.evaluation = dict(
        interval=1, 
        start=1,
        #save_best='auto' => get acc
        metric = 'mIoU',
        save_best = 'mIoU',
        pre_eval = True
    )
    cfg.optimizer = dict(
            constructor='LearningRateDecayOptimizerConstructor',
            type='AdamW',
            lr=0.00008,
            betas=(0.9, 0.999),
            weight_decay=0.05,
            paramwise_cfg={
                'decay_rate': 0.9,
                'decay_type': 'stage_wise',
                'num_layers': 12
        })
    
    cfg.lr_config = dict(
            policy='CosineRestart', 
            periods=[ 2*(2617 // cfg.data.samples_per_gpu + 1) for _ in range(200)],
            restart_weights=[1 for _ in range(200)],
            by_epoch = False,
            min_lr=1e-07
        )
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)

    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            #dict(type='ImageDetection'),
            #dict(type='TensorboardLoggerHook')
            #dict(type='CustomSweepHook')
            dict(type='MMSegWandbHook', 
                 init_kwargs=dict(project='semantic segmentation', 
                                  entity='arislid', 
                                  name=f'{model_name}_{k_fold}'),
                 interval=100, 
                 log_checkpoint=False, 
                 log_checkpoint_metadata=True,
                 num_eval_images = 10
            )
    ])
    
    
    
    cfg.device = get_device()
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=200)
    #cfg.load_from = './work_dirs/dyhead/best_bbox_mAP_50_epoch_12.pth'
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]
    
    #print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_segmentor(cfg.model)
    model.init_weights()

    meta = dict()
    #meta['fp16'] = dict(loss_scale=dict(init_scale=512))
    
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {False}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    logger.info(f'Set random seed to {cfg.seed}, '
                f'deterministic: {None}')
    
    logger.info(model)

    # 모델 학습
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, timestamp=timestamp, meta=meta)

if __name__ == '__main__':
    if selfos == 'Windows':
        freeze_support()
    #wandb.init(entity="revanZX",project="TrashSeg",name='conv_tiny')
    train(0)
    # cfg = Config.fromfile(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')
    # print(cfg.checkpoint_config)
    