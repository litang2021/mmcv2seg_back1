import copy
import warnings
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numbers
from numbers import Number

import cv2
import mmcv
import numpy as np
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.utils import is_tuple_of
from numpy import random
from scipy.ndimage import gaussian_filter

from mmseg.datasets.dataset_wrappers import MultiImageMixDataset
from mmseg.registry import TRANSFORMS




@TRANSFORMS.register_module()
class ColorJitter(BaseTransform):
    """Randomly change the brightness, contrast and saturation of an image.

    Modified from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/transforms.py
    Licensed under the BSD 3-Clause License.

    **Required Keys:**

    - img

    **Modified Keys:**

    - img

    Args:
        brightness (float | Sequence[float] (min, max)): How much to jitter
            brightness. brightness_factor is chosen uniformly from
            ``[max(0, 1 - brightness), 1 + brightness]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        contrast (float | Sequence[float] (min, max)): How much to jitter
            contrast. contrast_factor is chosen uniformly from
            ``[max(0, 1 - contrast), 1 + contrast]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        saturation (float | Sequence[float] (min, max)): How much to jitter
            saturation. saturation_factor is chosen uniformly from
            ``[max(0, 1 - saturation), 1 + saturation]`` or the given
            ``[min, max]``. Should be non negative numbers. Defaults to 0.
        hue (float | Sequence[float] (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from ``[-hue, hue]`` (0 <= hue
            <= 0.5) or the given ``[min, max]`` (-0.5 <= min <= max <= 0.5).
            Defaults to 0.
    """

    def __init__(self,
                 brightness: Union[float, Sequence[float]] = 0.,
                 contrast: Union[float, Sequence[float]] = 0.,
                 saturation: Union[float, Sequence[float]] = 0.,
                 hue: Union[float, Sequence[float]] = 0.):
        self.brightness = self._set_range(brightness, 'brightness')
        self.contrast = self._set_range(contrast, 'contrast')
        self.saturation = self._set_range(saturation, 'saturation')
        self.hue = self._set_range(hue, 'hue', center=0, bound=(-0.5, 0.5))

    def _set_range(self, value, name, center=1, bound=(0, float('inf'))):
        """Set the range of magnitudes."""
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    f'If {name} is a single number, it must be non negative.')
            value = (center - float(value), center + float(value))

        if isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                value = np.clip(value, bound[0], bound[1])
                from mmengine.logging import MMLogger
                logger = MMLogger.get_current_instance()
                logger.warning(f'ColorJitter {name} values exceed the bound '
                               f'{bound}, clipped to the bound.')
        else:
            raise TypeError(f'{name} should be a single number '
                            'or a list/tuple with length 2.')

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        else:
            value = tuple(value)

        return value

    @cache_randomness
    def _rand_params(self):
        """Get random parameters including magnitudes and indices of
        transforms."""
        trans_inds = np.random.permutation(4)
        b, c, s, h = (None, ) * 4

        if self.brightness is not None:
            b = np.random.uniform(self.brightness[0], self.brightness[1])
        if self.contrast is not None:
            c = np.random.uniform(self.contrast[0], self.contrast[1])
        if self.saturation is not None:
            s = np.random.uniform(self.saturation[0], self.saturation[1])
        if self.hue is not None:
            h = np.random.uniform(self.hue[0], self.hue[1])

        return trans_inds, b, c, s, h

    def transform(self, results: Dict) -> Dict:
        """Transform function to resize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: ColorJitter results, 'img' key is updated in result dict.
        """
        img = results['img']
        trans_inds, brightness, contrast, saturation, hue = self._rand_params()

        for index in trans_inds:
            if index == 0 and brightness is not None:
                img = mmcv.adjust_brightness(img, brightness)
            elif index == 1 and contrast is not None:
                img = mmcv.adjust_contrast(img, contrast)
            elif index == 2 and saturation is not None:
                img = mmcv.adjust_color(img, alpha=saturation)
            elif index == 3 and hue is not None:
                img = mmcv.adjust_hue(img, hue)

        results['img'] = img
        return results

    def __repr__(self):
        """Print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(brightness={self.brightness}, '
        repr_str += f'contrast={self.contrast}, '
        repr_str += f'saturation={self.saturation}, '
        repr_str += f'hue={self.hue})'
        return repr_str
