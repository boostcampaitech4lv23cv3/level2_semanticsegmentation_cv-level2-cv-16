class Config:
    config='./configs/_custom_/mask2former_beitv2_adapter_large_896_80k_ade20k_ms.py'
    exp_name='mask2former_beitv2_adapter_large_ADE20K'
    work_dir=f'./work_dirs/{exp_name}'
    load_from='./pretrained/mask2former_beitv2_adapter_large_896_80k_ade20k.pth'
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
    device='cuda'