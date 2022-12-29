_base_ = [
    '../_base_/models/upernet_dinat.py', '../_base_/datasets/trash-base_fold0.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

model = dict(
    backbone=dict(
        type='DiNAT',
        embed_dim=192,
        mlp_ratio=2.0,
        depths=[3, 4, 18, 5],
        num_heads=[6, 12, 24, 48],
        kernel_size=7,
        drop_path_rate=0.3,
        dilations=[[1, 20, 1], [1, 5, 1, 10], [1, 2, 1, 3, 1, 4, 1, 5, 1, 2, 1, 3, 1, 4, 1, 5, 1, 5], [1, 2, 1, 2, 1]],
        pretrained=None,
    ),
    decode_head=dict(
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=100000), ################ Online Hard Example Mining (OHEM)
        in_channels=[192, 384, 768, 1536],
        num_classes=11
    ),
    auxiliary_head=dict(
        in_channels=768,
        num_classes=11
    ))

# AdamW optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
                     'head': dict(lr_mult=10.), ################ Different Learning Rate(LR) for Backbone and Heads
                     'rpb': dict(decay_mult=0.),
                     'norm': dict(decay_mult=0.),
                 }),)

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2)

# No mixed precision with float16 in DiNAT-L
#fp16 = None
#optimizer_config = dict(
#    type="Fp16OptimizerHook",
#    grad_clip=None,
#    coalesce=True,
#    bucket_size_mb=-1,
#)
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=2618, max_keep_ckpts=1)
evaluation = dict(interval=2618, metric='mIoU', save_best='mIoU')