class Config:
    #config='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/mmsegmentation/configs/ade20k/eva_mask2former_896_40k_coco164k2ade20k_ss_relpos_layerscale_9dec.py'
    config='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/mmsegmentation/configs/ade20k/eva_mask2former_896_20k_coco164k2ade20k_ss.py'
    #config='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/mmsegmentation/configs/coco_stuff164k/eva_mask2former_896_60k_cocostuff164k_ss.py'
    exp_name='EVA_mask2former'
    work_dir=f'./work_dirs/{exp_name}'
    load_from='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/mmsegmentation/configs/ade20k/eva_sem_seg_mask2former_ade_relpos_layerscale_9dec_ss61p5_ms62p3.pth'
    resume_from=None
    no_validate=None
    gpus=None
    gpu_ids=None
    seed=42
    deterministic=None
    options=None
    cfg_options=None
    launcher='none'
    local_rank=0
    auto_resume=None


    