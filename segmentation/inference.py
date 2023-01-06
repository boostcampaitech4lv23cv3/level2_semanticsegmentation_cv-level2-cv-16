import os
import argparse
import json

import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403
from mmcv import Config
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import single_gpu_test
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

import numpy as np
import pandas as pd
import albumentations as A


def parse_args():
    parser = argparse.ArgumentParser(description='Inference a segmentor')
    parser.add_argument(
        '--config', 
        default='/opt/ml/input/seg/segmentation/configs/_custom_/mask2former_beitv2_adapter_large_896_80k_cocostuff164k_ss.py',
        help='train config file path')
    parser.add_argument(
        '--img_path', 
        default='/opt/ml/input/data/test_images', 
        help='test images path')
    parser.add_argument(
        '--work_dir', 
        default='/opt/ml/input/seg/segmentation/work_dirs/beitv2_adapter_large_data_v1_fold1_test', 
        help='work_dir path')
    parser.add_argument(
        '--pth_name', 
        default='best_mIoU_iter_43000', 
        help='best pth name')
    parser.add_argument(
        '--aug_test', 
        type=bool,
        default=False, 
        help='Use Flip and Multi scale aug')
    parser.add_argument(
        '--json_dir', 
        default='/opt/ml/input/data/test.json', 
        help='test json path')
    parser.add_argument(
        '--submission', 
        default='/opt/ml/input/data/sample_submission.csv', 
        help='submission path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    # dataset config 수정
    cfg.data.test.img_dir = args.img_path
    # cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize
    cfg.data.test.test_mode = True
    cfg.data.samples_per_gpu = 1
    cfg.work_dir = args.work_dir
    # cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.model.train_cfg = None
    # checkpoint path
    checkpoint_path = os.path.join(cfg.work_dir, f'{args.pth_name}.pth')

    if args.aug_test:
        # hard code index
        # cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True
        cfg.data.test.pipeline[1].flip_direction = ['horizontal', 'vertical']
        
    dataset = build_dataset(cfg.data.test)
    img_len = len(os.listdir(args.img_path))
    if len(dataset) != img_len: # 819
        raise AssertionError(f'Not match dataset images length.')
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    
    model = MMDataParallel(model.cuda(), device_ids=[0])
    output = single_gpu_test(model, data_loader)
    
    # sample_submisson.csv 열기
    submission = pd.read_csv(args.submission, index_col=None)
    with open(args.json_dir, "r", encoding="utf8") as outfile:
        datas = json.load(outfile)

    # set resize
    # input_size = 512
    output_size = 256
    transformed = A.Compose([A.Resize(output_size, output_size)])

    # PredictionString 대입
    for image_id, predict in enumerate(output):
        image_id = datas["images"][image_id]
        file_name = image_id["file_name"]
        temp_mask = []
        mask = np.array(predict, dtype='uint8')
        mask = transformed(image=mask)
        temp_mask.append(mask['image'])
        oms = np.array(temp_mask)
        oms = oms.reshape([oms.shape[0], output_size*output_size]).astype(int)

        string = oms.flatten()
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, 
                                    ignore_index=True)

    # submission.csv로 저장
    submission.to_csv(os.path.join(cfg.work_dir, f'submission_{args.pth_name}.csv'), index=False)

    print('Done')


if __name__ == '__main__':
    main()