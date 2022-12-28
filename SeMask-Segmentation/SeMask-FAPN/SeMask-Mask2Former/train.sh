#!/bin/sh

CUDA_VISIBLE_DEVICES=0 python train_net.py --dist-url 'tcp://127.0.0.1:50162' --num-gpus 1 --config-file configs/ade20k/semantic-segmentation/semask_swin/custom_trash_semantic_segmentation.yaml