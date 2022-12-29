class Config:
    #config='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/Neighborhood-Attention-Transformer/segmentation/configs/dinat/upernet_dinat_base_512x512_160k_ade20k.py'
    config='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/Neighborhood-Attention-Transformer/segmentation/configs/dinat/upernet_dinat_large_640x640_160k_ade20k_2.py'
    exp_name='upernet_dinat'
    work_dir=f'./work_dirs/{exp_name}'
    #custom_imports = dict(imports=['.mmseg.datasets.trash'], allow_failed_imports=False)
    load_from='/opt/ml/input/code/level2_semanticsegmentation_cv-level2-cv-16/Neighborhood-Attention-Transformer/segmentation/configs/dinat/upernet_dinat_large.pth'
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


    