class Config:
    config='./configs/_custom_/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py'
    exp_name='beitv2_adapter_large_data_v1_fold2'
    work_dir=f'./work_dirs/{exp_name}_1'
    wandb_project='Trash_Segmentation'
    wandb_entity='youngjun04'
    load_from=None
    resume_from='/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-16/segmentation/work_dirs/beitv2_adapter_large_data_v1_fold2/iter_43000.pth'
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
