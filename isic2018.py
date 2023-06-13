from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class ISIC2018(BaseSegDataset):
  classes = ('background', 'feature1', 'feature2', 'feature3', 'feature4', 'feature5')
  # palette = [[128, 128, 128], [151, 189, 8]]
  palette = [[128, 64, 128],  [70, 70, 70], 
                 [190, 153, 153], [250, 170,30], 
                 [107, 142, 35],  [70, 130, 180]]

  METAINFO = dict(classes = classes, palette = palette)
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
    self.balanced = kwargs.get("balanced", False)
    