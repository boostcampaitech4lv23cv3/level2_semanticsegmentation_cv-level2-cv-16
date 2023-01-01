from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class CustomTrashDataset(CustomDataset):
    """COCO-Stuff dataset.
    In segmentation map annotation for COCO-Stuff, Train-IDs of the 10k version
    are from 1 to 171, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    # 'Background'
    CLASSES = [ 
        'General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
        'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing'
    ]

    # [  0,   0,   0]
    PALLETE = [[128,   0,   0], [  0, 128,   0], [128, 128,   0],
               [  0,   0, 128], [128,   0, 128], [  0, 128, 128], [128, 128, 128],
               [ 64,   0,   0], [192,   0,   0],[ 64, 128,   0]] # class_dict.csv 참고

    def __init__(self, **kwargs):
        super(CustomTrashDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)