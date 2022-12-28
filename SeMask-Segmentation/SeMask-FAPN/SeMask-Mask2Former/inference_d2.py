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


from mask2former import SemanticSegmentorWithTTA, add_maskformer2_config

selfos = platform.system()

root = '/opt/ml/input'
dataset_path = "data"

model_name = 'SeMask-FAPN'
config_dir = './configs/ade20k/semantic-segmentation/semask_swin'
config_name = 'custom_trash_semantic_segmentation.yaml'
pth_name = './output/model_0059999.pth'
k_fold = 0
n_iter = 60000


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
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = args.weight
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.freeze()
    
    return cfg

        
def main():
    args = get_parser()
    cfg = setup(args)  
    
    submission = pd.read_csv(os.path.join(root ,"submission/sample_submission.csv"), index_col=None)
    json_dir = os.path.join(root, dataset_path, "test.json")
    
    with open(json_dir) as f:
        test_files = json.load(f)
    
    images = test_files['images']
    predictor = DefaultPredictor(cfg)

    input_size = 512
    output_size = 256
    image_id = []
    preds_array = np.empty((0, output_size * output_size), dtype=np.compat.long)
    for index, image_info in enumerate(tqdm(images, total=len(images))):
        file_name = image_info['file_name']
        image_id.append(file_name)
        
        path = os.path.join(root, dataset_path, file_name)
        img = read_image(path, format="BGR")
        
        pred = predictor(img)
        output = pred['sem_seg'].argmax(dim=0).detach().cpu().numpy()
        temp_mask = []
        temp_img = output.reshape(1, 512, 512)
        mask = temp_img.reshape((1, output_size, input_size//output_size, output_size, input_size//output_size)).max(4).max(2)
        temp_mask.append(mask)

        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size * output_size]).astype(int)
        preds_array = np.vstack((preds_array, oms))



    # for file_name, string in zip(image_id, preds_array):
    #     submission = submission.append(
    #         {"image_id": file_name, "PredictionString": ' '.join(str(e) for e in string.tolist())},
    #         ignore_index=True)
    submission = pd.concat([submission, pd.DataFrame([{"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in preds_array.tolist())}])]
                            , ignore_index=True)
    submission.to_csv(os.path.join(root, 'submission', f'submission_{model_name}_iter{n_iter}_fold-{k_fold}.csv'), index=False)
    
      
if __name__ == "__main__":

    if selfos == 'Windows':
        freeze_support()
    main()


    
    