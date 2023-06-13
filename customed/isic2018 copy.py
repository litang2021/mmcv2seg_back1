import os.path as osp
import pickle

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmengine
from mmengine import fileio


@DATASETS.register_module()
class ISIC2018(BaseSegDataset):
    classes = ('background', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5')
    palette = [[128, 64, 128],  [70, 70, 70], 
              [190, 153, 153], [250, 170,30], 
              [107, 142, 35],  [70, 130, 180]]
    balance_cache_file = "dataloadtrain.cache"
    METAINFO = dict(classes = classes, palette = palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
        self.balanced = kwargs.get("balanced", False) if not self.test_mode else False
        self.balanced_init()

    def balanced_init(self):
        if self.balanced:
            with open(osp.join(self.data_root, self.balance_cache_file), 'rb') as f:
                cache = pickle.load(f)
            self.balance_len = cache["balance_len"]
            self.epoch_pool = cache["epoch_pool"]
            self.inputs_ = cache["inputs_"]
            self.labels_ = cache["labels_"]
         
    

    
    