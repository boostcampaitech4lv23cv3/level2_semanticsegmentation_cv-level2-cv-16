_base_ = [
    '../_base_/models/upernet_dinat.py', '../_base_/datasets/trash-base_fold0.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
laod_from = '/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/Neighborhood-Attention-Transformer/segmentation/configs/dinat/upernet_dinat_base.pth'
model = dict(
    backbone=dict(
        type='DiNAT',
        embed_dim=128,
        mlp_ratio=2.0,
        depths=[3, 4, 18, 5],
        num_heads=[4, 8, 16, 32],
        drop_path_rate=0.5,
        kernel_size=7,
        layer_scale=1e-5,
        dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
        pretrained='',
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=11
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=11
    ))

# AdamW optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={
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

# Mixed precision
fp16 = None
optimizer_config = dict(
    type="Fp16OptimizerHook",
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
)

runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=2618, max_keep_ckpts=1)
evaluation = dict(interval=2618, metric='mIoU', save_best='mIoU')