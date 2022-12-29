import os
import platform
from multiprocessing import freeze_support
import numpy as np

from tqdm import tqdm
import pandas as pd

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.apis import single_gpu_test

import json
import cv2
selfos = platform.system() 

dataset_path = "../../data"

project_dir = '../mmsegmentation'
model_dir = 'convnext_fb'
model_name = 'upernet_convnext_xlarge_640_160k_ade20k_ms'
work_dirs = 'work_dirs'
data_root = '../../data'
pth_name = 'best_mIoU_epoch_7'
k_fold = 0

CLASSES = [
        'Background', 'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
    ]

PALLETE = [[  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
           [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
           [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0]]


category_names = CLASSES

def write_csv(output,cfg):
    input_size = 512
    output_size = 256

    submission = pd.read_csv("../../submission/sample_submission.csv", index_col=None)
    json_dir = os.path.join(dataset_path, "test.json")


    with open(json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # PredictionString 대입
    for image_id, predict in enumerate(tqdm(output)):

        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]

        temp_mask = []
        predict = predict.reshape(1, 512, 512)
        mask = predict.reshape((1, output_size, input_size//output_size, output_size, input_size//output_size)).max(4).max(2) # resize to 256*256
        temp_mask.append(mask)
        oms = np.array(temp_mask)
        # cv2.imwrite(os.path.join('./mask_test', file_name), oms)
        oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)
        string = oms.flatten()
        submission = pd.concat([submission, pd.DataFrame([{"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}])]
                                       , ignore_index=True)

    submission.to_csv(os.path.join('../../submission', f'submission_{model_name}_{k_fold}.csv'), index=False)

def inference():
    cfg = Config.fromfile(f'{project_dir}/configs/_TrashSEG_/{model_dir}/{model_name}.py')
    cfg.data.test.img_dir = data_root + '/test'
    cfg.data.test.test_mode = True
    
    cfg.data.samples_per_gpu = 4
    cfg.seed=24
    
    cfg.model.train_cfg = None
    cfg.model.pretrained = None
    
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model,os.path.join(project_dir,work_dirs,model_name + '_0',pth_name+'.pth'), map_location='cpu')
    model.CLASSES = CLASSES
    model = MMDataParallel(model.cuda(), device_ids=[0])
    output = single_gpu_test(model, data_loader)
    write_csv(output, cfg)
    
if __name__ == '__main__':
    if selfos == 'Windows':
        freeze_support()
    #wandb.init(entity="revanZX",project="TrashSeg",name='conv_tiny')
    inference()