import os
import os.path as osp
import argparse
import shutil
import json
import copy
from math import ceil


def parse_args():
    parser = argparse.ArgumentParser(description='split coco format json')
    parser.add_argument(
        '--input_dir', 
        default='/opt/ml/input/data',
        help='image batch foler path')
    parser.add_argument(
        '--split', 
        default='train_all', 
        help='coco json name')
    parser.add_argument(
        '--split_count', 
        type=int,
        default=5, 
        help='coco json name')
    parser.add_argument(
        '--save_path', 
        default='/opt/ml/input/data/train_all_split', 
        help='work_dir path')
    args = parser.parse_args()
    return args


def create_json(data, batch, page, save_path):
    f_anno = []
    f_images = []
    img_ids = list(batch.keys())
    for anno in range(len(data['annotations'])):
        if data['annotations'][anno]['image_id'] in img_ids:
            f_anno.append(data['annotations'][anno])
            
    for i in range(len(data['images'])):
        if data['images'][i]['id'] in img_ids:
            f_images.append(data['images'][i])
            
    _data = copy.deepcopy(data)
    _data['images'] = f_images[:]
    _data['annotations'] = f_anno[:]
    
    with open(osp.join(save_path, f'train_all_fold{page}.json'), 'w') as f: 
        json.dump(_data, f, indent=4)


def main():
    args = parse_args()
    annotation_file = os.path.join(args.input_dir, '{}.json'.format(args.split))
    
    with open(annotation_file) as f: 
        data = json.load(f)
        
    img_n = ceil(len(data['images']) / args.split_count)
    batch_lst = []
    for batch in range(args.split_count):
        start = img_n * batch
        img_dict = {}
        try:
            for info in data['images'][start:start+img_n]:
                img_dict[info['id']] = info['file_name']
        except:
            for info in data['images'][start:]:
                img_dict[info['id']] = info['file_name']
        batch_lst.append(img_dict)
        
    for i in range(args.split_count):
        for img_id, file_name in batch_lst[i].items():
            img_origin_path = osp.join(args.input_dir, file_name)
            img_output_path = osp.join(args.save_path, f'split{i}', file_name)
            os.makedirs(osp.join(args.save_path, f'split{i}', file_name.split('/')[0]), exist_ok=True)
            shutil.copy(img_origin_path, img_output_path)
        create_json(data, batch_lst[i], i, args.save_path)
        print(f"create batch {i}")
    print("Done")


if __name__ == '__main__':
    main()