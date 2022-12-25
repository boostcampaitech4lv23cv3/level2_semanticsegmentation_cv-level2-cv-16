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
    with open(osp.join(args.json_dir, json_lst[0])) as f: 
        data = json.load(f)
    # merge json
    for path in json_lst[1:]:
        assert path.split('.')[-1] == 'json', f"{path} file is not json"
        with open(osp.join(args.json_dir, path)) as f: 
            _data = json.load(f)
        data['images'].extend(_data['images'])
        data['annotations'].extend(_data['annotations'])
    # sorting by id
    data['images'].sort(key=lambda x: x['id'])
    data['annotations'].sort(key=lambda x: x['id'])
    # save json
    with open(osp.join(args.save_path, f'merge_coco.json'), 'w') as f: 
        json.dump(data, f, indent=4)

if __name__ == '__main__':
    main()