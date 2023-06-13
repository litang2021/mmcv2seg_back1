import os.path as osp
import pickle
import random
import os

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
  def __init__(self, balanced=False, mode="oversample",length_streak=None,**kwargs):
    test_mode = kwargs.get("test_mode", False)
    self.data_root = kwargs["data_root"]
    self.balanced = balanced if not test_mode else False
    self.bal_oversample = mode.lower() == "oversample"
    self.length_streak = length_streak * 2 if length_streak else 0
    if mode.lower() == "oversample":
      self.mode = 0x01
    elif mode.lower() == "randomsample":
      self.mode = 0x00
    elif mode.lower() == "oversample2":
      self.mode = 0x11
    elif mode.lower() == "streaks":
      self.mode = 0x20
    elif mode.lower() == "oversample3":
      self.mode = 0x02
    elif mode.lower() == "oversample_ex":
      self.mode = 0x30
      self.img_train_ex = "data/img_dir/train_ex"
      self.ann_train_ex = "data/ann_dir/train_ex"
    else:
      self.mode = 0xFF
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
      if self.mode == 0x01:
        ret = self.load_data_list_balance()
      elif self.mode == 0x00:
        ret = self.load_data_list_balance_random()
      elif self.mode == 0x11:
        ret = self.load_data_list_balance2()
      elif self.mode == 0x20:
        ret = self.load_data_list_balance_streaks()
      elif self.mode == 0x02:
        ret = self.load_data_list_balance3()
      elif self.mode == 0x30:
        ret = self.load_data_list_balance4()
      random.shuffle(ret)
      return ret
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
      if i == 3:
        continue
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

    for idx in range(self.balance_len * 10 * (400 // 80)):
      idx_ = idx % len(self.epoch_pool[3])
      idx_in_origin = self.epoch_pool[3][idx_]
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



    # add  f3
    for idx in range(self.balance_len * 20):
      idx_ = idx % len(self.epoch_pool[3])
      idx_in_origin = self.epoch_pool[3][idx_]
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
  
  def load_data_list_balance3(self):
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
      if i == 3:
        continue
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
      idx_ = idx % len(self.epoch_pool[3])
      idx_in_origin = self.epoch_pool[3][idx_]
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
      

    return data_list
  
  def load_data_list_balance4(self):
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
    
    for _ in range(2):
      for train_name in os.listdir(self.img_train_ex):
        img_path = osp.join(self.img_train_ex, train_name)
        lbl_path = osp.join(self.ann_train_ex, train_name[:-3] + "png")
        data_info = dict(
            img_path = img_path,
            seg_map_path = lbl_path,
            label_map = self.label_map,
            reduce_zero_label = self.reduce_zero_label,
            seg_fields = []
          )
        data_list.append(data_info)

      

    return data_list
  

  def load_data_list_balance_streaks(self):
    data_list = []
    img_dir = self.data_prefix.get('img_path', None)
    ann_dir = self.data_prefix.get('seg_map_path', None)
    # for target_ in range(31):
    for i in range(self.length_streak):
      idx = i % 10
      if idx > 5:
        idx = 3
      max_idx = len(self.epoch_pool[idx])
      idx_in_origin = self.epoch_pool[idx][random.randint(0, max_idx - 1)]
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

    return data_list

  def load_data_list_balance_random(self):
    img_dir = self.data_prefix.get('img_path', None)
    ann_dir = self.data_prefix.get('seg_map_path', None)
    data_list = RandomGenerator(
      self.epoch_pool, ann_dir, img_dir, self.inputs_, self.labels_,
      self.label_map, self.reduce_zero_label)
    return data_list

  def load_data_list_balance2(self):
    img_dir = self.data_prefix.get('img_path', None)
    ann_dir = self.data_prefix.get('seg_map_path', None)
    data_list = RandomGenerator2(
      self.epoch_pool, ann_dir, img_dir, self.inputs_, self.labels_,
      self.label_map, self.reduce_zero_label)
    return data_list


@DATASETS.register_module()
class ISIC2018_F3(BaseSegDataset):
  classes = ('milia-like cysts', "background")
  palette = [[128, 128, 128], [0, 0, 0]]
  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(img_suffix=".jpg", seg_map_suffix='.png', *args, **kwargs)


class RandomGenerator:
  import random
  def __init__(self, pools, ann_dir, img_dir, inputs_, labels_, label_map, reduce_zero_label):
    self.pools = pools
    self.ann_dir = ann_dir
    self.img_dir = img_dir
    self.n_cls = len(pools)
    self.n = sum([len(pool) for pool in pools])
    self.m = self.n // self.n_cls
    self.inputs_ = inputs_
    self.labels_ = labels_
    self.label_map = label_map
    self.reduce_zero_label = reduce_zero_label
    
    # for run
    self.container = []
    for i in range(5000 * 64):
      self.container.append(self.__getitem(i))
    
  
  def __len__(self):
    return 5000 * 64
  def __getitem__(self, idx):
    return self.container[idx]
  
  def clear(self):
    self.container = []
  def __getitem(self, idx):
  # def __getitem__(self, idx):
    # cls_idx = random.randint(1, self.n_cls) - 1
    cls_idx = idx % (self.n_cls * 2)
    if cls_idx >= self.n_cls:
      cls_idx = 3
    max_index = len(self.pools[cls_idx])
    index = random.randint(1, max_index) - 1
    index = self.pools[cls_idx][index]
    img_path = osp.join(self.img_dir, self.inputs_[index])
    lbl_path = osp.join(self.ann_dir, self.labels_[index])

    data_info = dict(
        img_path = img_path,
        seg_map_path = lbl_path,
        label_map = self.label_map,
        reduce_zero_label = self.reduce_zero_label,
        seg_fields = []
      )
    return data_info


class RandomGenerator2:
  import random
  def __init__(self, pools, ann_dir, img_dir, inputs_, labels_, label_map, reduce_zero_label):
    self.pools = pools
    self.ann_dir = ann_dir
    self.img_dir = img_dir
    self.n_cls = len(pools)
    self.n = sum([len(pool) for pool in pools])
    self.m = self.n // self.n_cls
    self.inputs_ = inputs_
    self.labels_ = labels_
    self.label_map = label_map
    self.reduce_zero_label = reduce_zero_label
    
    # for run
    self.container = []
    for i in range(5000 * 64):
      self.container.append(self.__getitem(i))
    
  
  def __len__(self):
    return 5000 * 64
  def __getitem__(self, idx):
    return self.container[idx]
  
  def clear(self):
    self.container = []
  def __getitem(self, idx):
  # def __getitem__(self, idx):
    # cls_idx = random.randint(1, self.n_cls) - 1
    cls_idx = idx % 31
    if 0 < cls_idx <= 5:
      cls_idx = 1
    elif 5 < cls_idx <= 10:
      cls_idx = 2
    elif 10 < cls_idx <= 15:
      cls_idx = 3
    elif 15 < cls_idx <= 20:
      cls_idx = 4
    elif cls_idx > 20:
      cls_idx = 5
    max_index = len(self.pools[cls_idx])
    index = random.randint(1, max_index) - 1
    index = self.pools[cls_idx][index]
    img_path = osp.join(self.img_dir, self.inputs_[index])
    lbl_path = osp.join(self.ann_dir, self.labels_[index])

    data_info = dict(
        img_path = img_path,
        seg_map_path = lbl_path,
        label_map = self.label_map,
        reduce_zero_label = self.reduce_zero_label,
        seg_fields = []
      )
    return data_info
 
  
  








    
    