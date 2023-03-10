# 모듈 import
import platform
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.datasets import (build_dataloader, build_dataset)
from mmseg.utils import get_device
from multiprocessing import freeze_support

import wandb
import wandb_config

from mmcv.runner.hooks import HOOKS, Hook


selfos = platform.system() 

model_dir = 'convnext_fb'
model_name = 'upernet_convnext_xlarge_640_160k_ade20k_ms'
work_dir = f'./work_dirs/{model_name}'
data_root = '../../data'

def train():
    w_run = wandb.init(config=wandb_config.hyperparameter_defaults)
    w_config = wandb.config
    
    k_fold = w_config.k_fold

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')

    #get k_fold
    cfg.data.train.img_dir = data_root + f'/images/train_{k_fold}'
    cfg.data.train.ann_dir = data_root + f'/annotations/train_{k_fold}'
    cfg.data.val.img_dir   = data_root + f'/images/val_{k_fold}'
    cfg.data.val.ann_dir   = data_root + f'/annotations/val_{k_fold}'
    
    
    cfg.data.workers_per_gpu = 4 #num_workers
    cfg.data.samples_per_gpu = w_config.batch_size

    cfg.seed = w_config.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir+f'_{k_fold}_sweep'

    cfg.evaluation = dict(
        interval=1, 
        start=1,
        #save_best='auto' => get acc
        metric = 'mIoU',
        #save_best = 'mIoU',
        pre_eval = True
    )
    
    
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)
    
    optimizer_name = w_config.optimizer
    if optimizer_name == 'AdamW':
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
    elif optimizer_name == 'SGD':
        cfg.optimizer = dict(
            type='SGD', 
            lr = 0.01, 
            momentum=0.9, 
            weight_decay=0.0005
        )
    
    t_lr = cfg.optimizer.lr
    scheduler_name = w_config.scheduler
    if scheduler_name == 'poly':
        cfg.lr_config = dict(
            policy='poly',
            warmup='linear',
            warmup_iters=1500,
            warmup_ratio=1e-6,
            power=1.0,
            min_lr=0.0,
            by_epoch=False
        )
    elif scheduler_name == 'Step':
        cfg.lr_config = dict(
            policy='step',
            warmup='linear',
            warmup_iters=500,
            warmup_ratio=0.001,
            step=[16, 19])
        
    elif scheduler_name == 'Cosine':
        cfg.lr_config = dict(
            policy='CosineRestart', 
            periods=[ 2*(2617 // w_config.batch_size + 1) for _ in range(w_config.epochs)],
            restart_weights=[1 for _ in range(w_config.epochs)],
            by_epoch = False,
            min_lr=1e-07
        )
        
    cfg.checkpoint_config = dict(max_keep_ckpts=2, interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            #dict(type='ImageDetection'),
            #dict(type='TensorboardLoggerHook')
            #dict(type='CustomSweepHook')
            dict(type='MMSegWandbHook', 
                 #init_kwargs=dict(project='Trash_Seg', 
                 #                 entity='revanZX', 
                 #                 name=f'{model_name}_{k_fold}'),
                 interval=100, 
                 log_checkpoint=False, 
                 log_checkpoint_metadata=True,
                 #num_eval_image = 10
            )
    ])
    
    cfg.device = get_device()
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=w_config.epochs)
    #cfg.load_from = './work_dirs/dyhead/best_bbox_mAP_50_epoch_12.pth'
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]
    
    #print(datasets[0])

    # 모델 build 및 pretrained network 불러오기
    model = build_segmentor(cfg.model)
    model.init_weights()

    meta = dict()
    #meta['fp16'] = dict(loss_scale=dict(init_scale=512))

    # 모델 학습
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=meta)

if __name__ == '__main__':
    if selfos == 'Windows':
        freeze_support()
    
    sweep_id = wandb.sweep(wandb_config.sweep_config, project="sweep_convnext", entity="revanZX")
    wandb.agent(sweep_id, train, count=10)