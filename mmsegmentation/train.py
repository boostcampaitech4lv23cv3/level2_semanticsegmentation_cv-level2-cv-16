# 모듈 import
import platform
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
# from mmcv_custom import train_segmentor
from mmseg.datasets import (build_dataloader, build_dataset)
from mmseg.utils import get_device
from multiprocessing import freeze_support

import wandb
import wandb_config

from mmcv.runner.hooks import HOOKS, Hook


selfos = platform.system() 

model_dir = 'hornet'
model_name = 'upernet_hornet_large_gf_640_160k_ade20k'
work_dir = f'./work_dirs/{model_name}_v2_2'
data_root = '../../data/kfold_v2_no_area'

def train(k_fold):

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')

    #get k_fold
    cfg.data.train.img_dir = data_root + f'/images/train_{k_fold}'
    cfg.data.train.ann_dir = data_root + f'/annotations/train_{k_fold}'
    cfg.data.val.img_dir   = data_root + f'/images/val_{k_fold}'
    cfg.data.val.ann_dir   = data_root + f'/annotations/val_{k_fold}'
    
    cfg.data.workers_per_gpu = 4 #num_workers
    cfg.data.samples_per_gpu = 8

    cfg.seed = 24
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
    # cfg.optimizer = dict(
    #         constructor='LearningRateDecayOptimizerConstructor',
    #         type='AdamW',
    #         lr=0.00008,
    #         betas=(0.9, 0.999),
    #         weight_decay=0.05,
    #         paramwise_cfg={
    #             'decay_rate': 0.9,
    #             'decay_type': 'stage_wise',
    #             'num_layers': 12
    #     })
    
    # cfg.lr_config = dict(
    #         policy='CosineRestart', 
    #         periods=[ 2*(2617 // cfg.data.samples_per_gpu + 1) for _ in range(200)],
    #         restart_weights=[1 for _ in range(200)],
    #         by_epoch = False,
    #         min_lr=1e-07
    #     )
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)

    cfg.checkpoint_config = dict(max_keep_ckpts=1, interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            #dict(type='ImageDetection'),
            #dict(type='TensorboardLoggerHook')
            #dict(type='CustomSweepHook')
            dict(type='MMSegWandbHook', 
                 init_kwargs=dict(project='Trash_Segmentation', 
                                  entity='youngjun04', 
                                  name=f'{model_name}_v2_{k_fold}'),
                 interval=100, 
                 log_checkpoint=False, 
                 log_checkpoint_metadata=True,
                #  num_eval_images = 50
            )
    ])
    
    cfg.device = get_device()
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=200)
    cfg.load_from = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-16/mmsegmentation/configs/_TrashSEG_/hornet/upernet_hornet_large_gf.pth'
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
    #wandb.init(entity="revanZX",project="TrashSeg",name='conv_tiny')
    train(0)