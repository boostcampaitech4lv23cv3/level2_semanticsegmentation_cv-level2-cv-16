# 모듈 import
import platform
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
<<<<<<< HEAD
# from mmseg.apis import train_segmentor
from mmcv_custom import train_segmentor
from mmseg.datasets import (build_dataloader, build_dataset)
=======
from mmseg.apis import train_segmentor
>>>>>>> origin/T4073
from mmseg.utils import get_device
from multiprocessing import freeze_support

import wandb
import wandb_config

from mmcv.runner.hooks import HOOKS, Hook


selfos = platform.system() 

model_dir = 'beit_unlim'
model_name = 'upernet_beit_large_24_512_slide_160k_ade20k_pt2ft'
work_dir = f'./work_dirs/{model_name}'
data_root = '../../data'

def train(k_fold):

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_TrashSEG_/{model_dir}/{model_name}.py')

    #get k_fold
    cfg.data.train.img_dir = data_root + f'/images/train_{k_fold}'
    cfg.data.train.ann_dir = data_root + f'/annotations/train_{k_fold}'
    cfg.data.val.img_dir   = data_root + f'/images/val_{k_fold}'
    cfg.data.val.ann_dir   = data_root + f'/annotations/val_{k_fold}'
    
    cfg.data.workers_per_gpu = 4 #num_workers
    cfg.data.samples_per_gpu = 2

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
    
    
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)

    cfg.checkpoint_config = dict(None)#dict(max_keep_ckpts=3, interval=1)
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
                                  name=f'{model_name}_{k_fold}'),
                 interval=100, 
                 log_checkpoint=False, 
                 log_checkpoint_metadata=True,
<<<<<<< HEAD
                #  num_eval_images = 50
=======
                 num_eval_image = 10
>>>>>>> origin/T4073
            )
    ])
    
    cfg.device = get_device()
    cfg.runner = dict(type='EpochBasedRunner', max_epochs=200)
    cfg.load_from = './configs/_TrashSEG_/beit_unlim/beit_large_patch16_640_pt22k_ft22ktoade20k.pth'
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