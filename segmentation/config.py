class Config:
    config='./configs/_custom_/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py'
    exp_name='mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss'
    work_dir=f'./work_dirs/{exp_name}'
    load_from='./pretrained/mask2former_beit_adapter_large_480_40k_pascal_context_59.pth.tar'
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