import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import load_sem_seg

PALLETE = [[  0,   0,   0], [128,   0,   0], [  0, 128,   0], [128, 128,   0],
            [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
            [ 64,   0,   0], [192,   0,   0], [ 64, 128,   0]]
class_labels = {
    0: "Backgroud",
    1: "General trash",
    2: "Paper",
    3: "Paper pack",
    4: "Metal",
    5: "Glass",
    6: "Plastic",
    7: "Styrofoam",
    8: "Plastic bag",
    9: "Battery",
    10: "Clothing",
}

def _get_trash_stuff_meta():
    stuff_ids = [k for k in class_labels.keys()]
    stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
    stuff_classes = [k for k in class_labels.values()]
    stuff_colors = [list(P) for P in PALLETE]
    
    ret = {
        "stuff_dataset_id_to_contiguous_id": stuff_dataset_id_to_contiguous_id,
        "stuff_classes": stuff_classes,
        "stuff_colors": stuff_colors,
    }
    
    return ret

def register_all_trash_full():
    root = f'/opt/ml/input/data'
    # root = os.path.join('/opt/ml/input/data', 'images')
    fold = 0
    meta = _get_trash_stuff_meta()
    for name, dirname in [("train", f"train_{fold}"), ("val", f"val_{fold}")]:
        image_dir = os.path.join(root, 'images', dirname)
        gt_dir = os.path.join(root, 'annotations', dirname)
        name = f"trash_recycle_sem_seg_{name}_{fold}" # train_recycle_sem_seg_train_0
        DatasetCatalog.register(
            name, lambda x=image_dir, y=gt_dir: load_sem_seg(y, x, gt_ext="png", image_ext="jpg")
        )
        MetadataCatalog.get(name).set(
            image_root=image_dir,
            sem_seg_root=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta
        )
    
if __name__=="__main__":
    path =os.path.join('/opt/ml/input/data', 'images', 'train_0') 
    # print(os.path.join('/opt/ml/input/data', 'images', 'train_0'))
    print(len(os.listdir(path)))
    # print([tuple(P) for P in PALLETE])
    register_all_trash_full()
    # print(MetadataCatalog.list())
    print(MetadataCatalog.get("trash_recycle_sem_seg_train_0").name)
    # print(MetadataCatalog.get("ade20k_sem_seg_train").stuff_classes)