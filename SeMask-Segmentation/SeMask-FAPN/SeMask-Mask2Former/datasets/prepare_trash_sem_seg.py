#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
from pathlib import Path

import numpy as np
import tqdm
from PIL import Image
from icecream import ic


def convert(input, output):
    img = np.asarray(Image.open(input))
    assert img.dtype == np.uint8
    img = img - 1  # 0 (ignore) becomes 255. others are shifted by 1
    Image.fromarray(img).save(output)

# No RGB information in annotations_detectron2...
# How to get RGB info from annotations images?
if __name__ == "__main__":
    dataset_dir = Path(os.getenv("DETECTRON2_DATASETS", "/opt/ml/input/data"))
    for name in ["train", "val"]:
        for fold in range(5):
            fold_dir = f'{name}_{fold}'
            annotation_dir = dataset_dir / "annotations" / fold_dir
            output_dir = dataset_dir / "annotations_detectron2" / fold_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            for file in tqdm.tqdm(list(annotation_dir.iterdir())):
                output_file = output_dir / file.name
                convert(file, output_file)
    # ic(dataset_dir)