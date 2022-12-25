# Copyright (c) OpenMMLab. All rights reserved.

# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = '/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/Neighborhood-Attention-Transformer/segmentation/configs/dinat/upernet_dinat_large.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
