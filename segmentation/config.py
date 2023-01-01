class Config:
    config='./configs/_custom_/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py'
    exp_name='beitv2_adapter_large_data_v1_fold0_albu_1'
    work_dir=f'./work_dirs/{exp_name}'
    wandb_project='Trash_Seg'
    wandb_entity='jjhun1228'
    load_from='./weights/mask2former_beitv2_adapter_large_896_80k_cocostuff164k.pth'
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
