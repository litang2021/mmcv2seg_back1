import os.path as osp
import pickle
import random

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset
import mmengine
from mmengine import fileio


@DATASETS.register_module()
class ISIC2018(BaseSegDataset):
  classes = ('background', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5')
  # palette = [[128, 128, 128], [151, 189, 8]]
  palette = [[128, 64, 128],  [70, 70, 70], 
                  [190, 153, 153], [250, 170,30], 
                  [107, 142, 35],  [70, 130, 180]]
  balance_cache_file = "dataloadtrain.cache"
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, balanced=False, **kwargs):
    test_mode = kwargs.get("test_mode", False)
    self.data_root = kwargs["data_root"]
    self.balanced = balanced if not test_mode else False
    self.balanced_init()
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', **kwargs)
    
  
  def balanced_init(self):
    if self.balanced:
      with open(osp.join(self.data_root, self.balance_cache_file), 'rb') as f:
          cache = pickle.load(f)
      self.balance_len = cache["balance_len"]
      self.epoch_pool = cache["epoch_pool"]
      self.inputs_ = cache["inputs_"]
      self.labels_ = cache["labels_"]
    
            
  def load_data_list(self):
    if self.balanced:
      return self.load_data_list_balance()
    else:
      return super().load_data_list()

  def load_data_list_balance(self):
    data_list = []
    img_dir = self.data_prefix.get('img_path', None)
    ann_dir = self.data_prefix.get('seg_map_path', None)
    # for target_ in range(31):
    for idx in range(self.balance_len):
      idx_in_origin = self.epoch_pool[0][idx]
      img_path = osp.join(img_dir, self.inputs_[idx_in_origin])
      lbl_path = osp.join(ann_dir, self.labels_[idx_in_origin])

      data_info = dict(
          img_path = img_path,
          seg_map_path = lbl_path,
          label_map = self.label_map,
          reduce_zero_label = self.reduce_zero_label,
          seg_fields = []
        )
      data_list.append(data_info)

    for i in range(1, 5):  # 1 2 3 4
      for idx in range(self.balance_len * 5):
      # for idx in range(self.balance_len * 5 if i !=3 else self.balance_len * 30):
        idx_ = idx % len(self.epoch_pool[i])
        idx_in_origin = self.epoch_pool[i][idx_]
        img_path = osp.join(img_dir, self.inputs_[idx_in_origin])
        lbl_path = osp.join(ann_dir, self.labels_[idx_in_origin])

        data_info = dict(
          img_path = img_path,
          seg_map_path = lbl_path,
          label_map = self.label_map,
          reduce_zero_label = self.reduce_zero_label,
          seg_fields = []
        )
        data_list.append(data_info)

    for idx in range(self.balance_len * 10):
      idx_ = idx % len(self.epoch_pool[5])
      idx_in_origin = self.epoch_pool[5][idx_]
      img_path = osp.join(img_dir, self.inputs_[idx_in_origin])
      lbl_path = osp.join(ann_dir, self.labels_[idx_in_origin])

      data_info = dict(
          img_path = img_path,
          seg_map_path = lbl_path,
          label_map = self.label_map,
          reduce_zero_label = self.reduce_zero_label,
          seg_fields = []
        )
      data_list.append(data_info)
      
    # for target in range(len(self.classes)):
    #   for idx in range(self.balance_len):
    #     idx_in_class = idx % len(self.epoch_pool[target])
    #     idx_in_origin = self.epoch_pool[target][idx_in_class]

    #     img_path = osp.join(img_dir, self.inputs_[idx_in_origin])
    #     lbl_path = osp.join(ann_dir, self.labels_[idx_in_origin])

    #     data_info = dict(
    #       img_path = img_path,
    #       seg_map_path = lbl_path,
    #       label_map = self.label_map,
    #       reduce_zero_label = self.reduce_zero_label,
    #       seg_fields = []
    #     )
    #     data_list.append(data_info)
    return data_list
  




    
    