# dataset settings
dataset_type = 'CustomTrashDataset'
data_root = '/opt/ml/input/data/kfold_v1'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
albu = [
        dict(
            type='OneOf',
            transforms=[
                dict(type='Flip',p=1.0),
                dict(type='RandomRotate90',p=1.0)
            ],
            p=0.5),
        dict(
            type='OneOf',
            transforms=[
                dict(
                    type='RandomBrightnessContrast',
                    brightness_limit=(-0.1, 0.15),
                    contrast_limit=(-0.1, 0.15),
                    p=1.0),
                dict(
                    type='CLAHE',
                    clip_limit=(2, 6),
                    tile_grid_size=(8, 8),
                    p=1.0),
            ],
            p=0.5),
    ]
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='Albu', 
    #     transforms=albu,
    #     keymap=dict(img="img", gt_semantic_seg="gt_semantic_seg"),
    #     update_pad_shape=False),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/train_0',
        ann_dir='annotations/train_0',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val_0',
        ann_dir='annotations/val_0',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images/val_0',
        ann_dir='annotations/val_0',
        pipeline=test_pipeline))
