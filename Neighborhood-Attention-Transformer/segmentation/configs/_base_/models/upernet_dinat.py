# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='DiNAT',
        embed_dim=64,
        mlp_ratio=3.0,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        drop_path_rate=0.2,
        kernel_size=7,
        dilations=[[1, 16, 1], [1, 4, 1, 8], [1, 2, 1, 3, 1, 4], [1, 2, 1, 2, 1]],
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        in_patch_size=4,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ignore_index=0, ################# Ignore specified label index in loss calculation
        loss_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[2., 4., 2.4, 3.4, 3.2, 3.8, 3.6, 2.6, 2.2, 2.8, 3.]), ################# Class Balanced Loss, Multiple Losses
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]
            ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=11,
        norm_cfg=norm_cfg,
        align_corners=False,
        # ignore_index=0, ################# Ignore specified label index in loss calculation
        loss_decode=[dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.2)]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
