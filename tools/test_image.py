"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""

from PIL import Image
import cv2
import argparse
import os
import numpy as np
import tqdm

import json

import shutil
data_root = '../../data'
mk_path = data_root + '/test'
def read_json(path):
    with open(path, 'r') as f:
        dataset = json.loads(f.read())
    return dataset

def main():
    test_data = read_json(os.path.join(data_root, 'test.json'))
    for imgs in  test_data['images']:
        img_name = imgs['file_name']
        img_path = os.path.join(data_root, img_name)
        
        shutil.copyfile(img_path, os.path.join(mk_path, img_name))
if __name__ == '__main__':
    main()