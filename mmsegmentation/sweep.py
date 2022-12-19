# 모듈 import
import platform
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor
from mmseg.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmseg.utils import get_device
from multiprocessing import freeze_support

import wandb
import wandb_config

from mmcv.runner.hooks import HOOKS, Hook


selfos = platform.system() 

model_dir = 'convnext'
model_name = 'upernet_convnext_tiny_fp16_512x512_160k_ade20k'
work_dir = f'./work_dirs/{model_name}'


@HOOKS.register_module()
class CustomSweepHook(Hook):
    """Check invalid loss hook.
    This hook will regularly check whether the loss is valid
    during training.
    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval=50):
        self.interval = interval
        
    def after_train_epoch(self, runner): # not finished
        wandb.log({'epoch': runner.epoch, "loss": runner.loss})

def train():
    w_run = wandb.init(config=wandb_config.hyperparameter_defaults)
    w_config = wandb.config
    
    k_fold = w_config.k_fold
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
               "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile(f'./configs/_trashDet_/{model_dir}/{model_name}.py')

    root='../../dataset/'
    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + f'train_{k_fold}.json' # train json 정보
    
    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = root
    cfg.data.val.ann_file = root + f'val_{k_fold}.json'
   
    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    
    cfg.train_pipeline = cfg.train_pipeline
    cfg.val_pipeline = cfg.test_pipeline
    cfg.test_pipeline = cfg.test_pipeline

    cfg.data.train.pipeline = cfg.train_pipeline
    cfg.data.val.pipeline = cfg.val_pipeline
    cfg.data.test.pipeline = cfg.test_pipeline

    cfg.data.workers_per_gpu = 8 #num_workers
    cfg.data.samples_per_gpu = w_config.batch_size #batch_size

    cfg.seed = w_config.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = work_dir + f'_{k_fold}'

    cfg.evaluation = dict(
        interval=1, 
        start=1,
        save_best='auto' 
    )
    
    
    cfg.optimizer_config.grad_clip = None #dict(max_norm=35, norm_type=2)
    #cfg.optimizer.lr=0.000005
    #cfg.lr_config.step=[4]
    
    scheduler_name = w_config.optimizer
    if scheduler_name == 'AdamW':
        cfg.optimizer = dict(
            constructor='LearningRateDecayOptimizerConstructor',
            _delete_=True,
            type='AdamW',
            lr=0.0001,
            betas=(0.9, 0.999),
            weight_decay=0.05,
            paramwise_cfg={
                'decay_rate': 0.9,
                'decay_type': 'stage_wise',
                'num_layers': 6
        })
    elif scheduler_name == 'SGD':
        cfg.optimizer = dict(
            _delete_=True, 
            type='SGD', 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=0.0005
        )
    
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.log_config = dict(
        interval=50,
        hooks=[
            dict(type='TextLoggerHook'),
            #dict(type='ImageDetection'),
            #dict(type='TensorboardLoggerHook')
            dict(type='CustomSweepHook')
            #dict(type='MMSegWandbHook', by_epoch=False, # The Wandb logger is also supported, It requires `wandb` to be installed.
            #     init_kwargs={'entity': "revanZX", # The entity used to log on Wandb
            #                  'project': "MMSeg", # Project name in WandB
            #                  'config': cfg_dict}), # Check https://docs.wandb.ai/ref/python/init for more init arguments.
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