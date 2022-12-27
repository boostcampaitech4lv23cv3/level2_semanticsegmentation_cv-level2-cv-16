import os
import os.path as osp
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='merge coco format json')
    parser.add_argument(
        '--json_dir',
        default='/opt/ml/input/data/merge_json', 
        help='merge json path')
    parser.add_argument(
        '--save_path', 
        default='/opt/ml/input/data', 
        help='work_dir path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    json_lst = os.listdir(args.json_dir)
    assert len(json_lst) > 1, "files count least more than 2"
    json_lst.sort()
    with open(osp.join(args.json_dir, json_lst[0])) as f: 
        data = json.load(f)
        
    img_idx = 0
    for i in range(len(data['images'])):
        data['images'][i]['id'] = img_idx
        img_idx += 1
    anno_idx = 0
    for i in range(len(data['annotations'])):
        data['annotations'][i]['id'] = anno_idx
        data['annotations'][i]['image_id'] -= 1
        anno_idx += 1
    # merge json
    img_pre = len(data['images']) - 1
    # img_idx = len(data['images'])+1
    # anno_idx = len(data['annotations'])+1
    for path in json_lst[1:]:
        assert path.split('.')[-1] == 'json', f"{path} file is not json"
        with open(osp.join(args.json_dir, path)) as f: 
            _data = json.load(f)
        for i in range(len(_data['images'])):
            _data['images'][i]['id'] = img_idx
            img_idx += 1
        for i in range(len(_data['annotations'])):
            _data['annotations'][i]['id'] = anno_idx
            _data['annotations'][i]['image_id'] += img_pre
            anno_idx += 1
        data['images'].extend(_data['images'])
        data['annotations'].extend(_data['annotations'])
        img_pre += len(_data['images'])
    # sorting by id
    # data['images'].sort(key=lambda x: x['id'])
    # data['annotations'].sort(key=lambda x: x['id'])
    # idx = 1
    # for i in range(len(data['images'])):
    #     data['images'][i]['id'] = idx
    #     idx += 1
        
    # idx = 1
    # for i in range(len(data['annotations'])):
    #     data['annotations'][i]['id'] = idx
    #     idx += 1
    
    # save json
    with open(osp.join(args.save_path, f'merge_coco.json'), 'w') as f: 
        json.dump(data, f)

if __name__ == '__main__':
    main()