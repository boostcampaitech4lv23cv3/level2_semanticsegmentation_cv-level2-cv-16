import copy
import itertools
import logging
from pathlib import Path
import os
import platform
from multiprocessing import freeze_support

from tqdm import tqdm
import pandas as pd
import numpy as np

import json
import cv2
import argparse


from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import ColorMode, Visualizer


from mask2former import SemanticSegmentorWithTTA, add_maskformer2_config, add_best_mIoU_checkpointer_config
from register_trash_dataset import register_all_trash_full
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

selfos = platform.system()

make_pseudo_images = True

root = '/opt/ml/input'
dataset_path = "data"
debug_root = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-16/SeMask-Segmentation'
model_name = 'SeMask-FAPN/SeMask-Mask2Former'
config_dir = './configs/ade20k/semantic-segmentation/semask_swin'
config_name = 'custom_trash_semantic_segmentation.yaml'
n_iter = 79999
pth_name = f'./trash_dataV1_WarmupPolyLR_1e-5/model_best_27299iter.pth'
k_fold = 0


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', type=str, default=f'{config_dir}/{config_name}')
    parser.add_argument('--weight', type=str, default=pth_name)
    arg = parser.parse_args()
    return arg

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    add_best_mIoU_checkpointer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()
    
    return cfg

        
def main():
    args = get_parser()
    cfg = setup(args)  
    register_all_trash_full()
    trash_metadata = MetadataCatalog.get("trash_recycle_sem_seg_train_0")

    json_dir = os.path.join(root, dataset_path, "test.json")
    with open(json_dir) as f:
        test_files = json.load(f)
    
    images = test_files['images']
    predictor = DefaultPredictor(cfg)

    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        path = os.path.join(root, dataset_path, file_name)
        # img = read_image(path, format="BGR")
        img = cv2.imread(path)
        
        pred = predictor(img)
        v = Visualizer(img[:, :, ::-1],
                       metadata=trash_metadata,
                       scale=1,
                       instance_mode=ColorMode.SEGMENTATION)
        output = v.draw_sem_seg(pred["sem_seg"].argmax(dim=0).detach().to("cpu").numpy())
        if not os.path.exists(os.path.join(root, dataset_path, 'pseudo_annotations')):
            os.mkdir(os.path.join(root, dataset_path, 'pseudo_annotations'))
        img_output = output.get_image()[:,:,::-1]
        cv2.imwrite(os.path.join(root, dataset_path, 'pseudo_annotations', f'pseudo_{index:04}.png'), img_output)


def main2():
    args = get_parser()
    cfg = setup(args)
    with open('/opt/ml/input/data/test.json') as f:
        test_files = json.load(f)
    images = test_files['images']
    predictor = DefaultPredictor(cfg)
    pseudo_dir = '/opt/ml/input/data/mmseg_remasking/annotations/test'
    os.makedirs(pseudo_dir)
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        path = Path('/opt/ml/input/data') / file_name
        img = read_image(path, format="BGR")
        pred = predictor(img)
        output = pred['sem_seg'].argmax(dim=0).detach().cpu().numpy()
        cv2.imwrite(os.path.join(pseudo_dir, str(index).zfill(4)+'.png'), output)
      
if __name__ == "__main__":
    
    if selfos == 'Windows':
        freeze_support()
    main()
    # register_all_trash_full()
    # trash_metadata = MetadataCatalog.get("trash_recycle_sem_seg_train_0")
    # print(trash_metadata.stuff_colors)
    # print(x/255 for x in trash_metadata.stuff_colors)
    # print(trash_metadata.stuff_colors[0] + 30)
    # mask_color = [x / 255 for x in trash_metadata.stuff_colors[label]]

    
    
