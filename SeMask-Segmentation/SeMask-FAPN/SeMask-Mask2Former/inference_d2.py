import copy
import itertools
import logging
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


from mask2former import SemanticSegmentorWithTTA, add_maskformer2_config, add_best_mIoU_checkpointer_config

selfos = platform.system()


root = '/opt/ml/input'
dataset_path = "data"
debug_root = '/opt/ml/input/level2_semanticsegmentation_cv-level2-cv-16/SeMask-Segmentation'
tools_root = 'level2_semanticsegmentation_cv-level2-cv-16/tools'
model_name = 'SeMask-FAPN'
config_dir = './configs/ade20k/semantic-segmentation/semask_swin'
config_name = 'custom_trash_semantic_segmentation.yaml'
pth_name = './trash_dataV1_WarmupPolyLR_1e-5/model_best_39000iter.pth'
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

        
def main(make_for_pseudo):
    args = get_parser()
    cfg = setup(args)  
    
    submission = pd.read_csv(os.path.join(root ,"submission/sample_submission.csv"), index_col=None)
    json_dir = os.path.join(root, dataset_path, "test.json")
    save_split_name = pth_name.split('/')
    save_name = f"{save_split_name[1]}-{save_split_name[2].split('.')[0]}"
    
    with open(json_dir) as f:
        test_files = json.load(f)
    
    images = test_files['images']
    predictor = DefaultPredictor(cfg)

    input_size = 512
    output_size = 256
    # image_id = []
    # preds_array = np.empty((0, output_size * output_size), dtype=np.long)
    # preds_array = np.empty((0, input_size * input_size), dtype=np.long)
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        # image_id.append(file_name)
        
        path = os.path.join(root, dataset_path, file_name)
        img = read_image(path, format="BGR")
        
        pred = predictor(img)
        output = pred['sem_seg'].argmax(dim=0).detach().cpu().numpy()
        temp_mask = []
        temp_img = output.reshape(1, 512, 512)
        
        if make_for_pseudo:
            mask = temp_img.reshape((1, input_size, input_size//input_size, input_size, input_size//input_size)).max(4).max(2)
            temp_mask.append(mask)
            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], input_size * input_size]).astype(int)
        else:
            mask = temp_img.reshape((1, output_size, input_size//output_size, output_size, input_size//output_size)).max(4).max(2)
            temp_mask.append(mask)
            oms = np.array(temp_mask)
            oms = oms.reshape([oms.shape[0], output_size * output_size]).astype(int)
        # preds_array = np.vstack((preds_array, oms))
 
        string = oms.flatten()
        submission = pd.concat([submission, pd.DataFrame([{"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}])]
                               , ignore_index=True)
        
    
    if make_for_pseudo:
        submission.to_csv(os.path.join(root, tools_root, f'make_for_pseudo_{model_name}_{save_name}_fold-{k_fold}.csv'), index=False)
    else:
        submission.to_csv(os.path.join(root, 'submission', f'submission_{model_name}_{save_name}_fold-{k_fold}.csv'), index=False)

    
      
if __name__ == "__main__":
    
    if selfos == 'Windows':
        freeze_support()
    make_for_pseudo = False
    main(make_for_pseudo)
    # print(len(preds_array[0]))
    # print(file_name[0])
    # save_split_name = pth_name.split('/')
    # save_name = f"{save_split_name[1]}-{save_split_name[2].split('.')[0]}"
    # print(save_name)
    


    
    