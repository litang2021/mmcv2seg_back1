# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmengine.model import BaseModule
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from mmseg.registry import MODELS
from mmseg.models.utils import ResLayer
from mmseg.models.backbones.resnet import BasicBlock, Bottleneck, ResNetV1c

from mmseg.registry import MODELS
from mmseg.models.utils import Upsample, resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmcv.cnn import ConvModule
import numpy as np
from einops import rearrange

from mmseg.models.decode_heads.point_head import (
    calculate_uncertainty, point_sample, BaseCascadeDecodeHead,
    SampleList, List, accuracy
    )

# from loguru import logger

@MODELS.register_module()
class ResNetV1c_5Out(ResNetV1c):
    def forward(self, x):
        """Forward function."""
        outs = []
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        outs.append(x) # 1/2
        x = self.maxpool(x)
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

@MODELS.register_module()
class ResNetnoPool(BaseModule):
    """ResNet backbone.

    This backbone is the improved implementation of `Deep Residual Learning
    for Image Recognition <https://arxiv.org/abs/1512.03385>`_.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Default: 3.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        num_stages (int): Resnet stages, normally 4. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
            Default: (1, 2, 2, 2).
        dilations (Sequence[int]): Dilation of each stage.
            Default: (1, 1, 1, 1).
        out_indices (Sequence[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer. Default: 'pytorch'.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters. Default: -1.
        conv_cfg (dict | None): Dictionary to construct and config conv layer.
            When conv_cfg is None, cfg will be set to dict(type='Conv2d').
            Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True).
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (dict | None): Dictionary to construct and config DCN conv layer.
            When dcn is not None, conv_cfg must be None. Default: None.
        stage_with_dcn (Sequence[bool]): Whether to set DCN conv for each
            stage. The length of stage_with_dcn is equal to num_stages.
            Default: (False, False, False, False).
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.

            - position (str, required): Position inside block to insert plugin,
            options: 'after_conv1', 'after_conv2', 'after_conv3'.

            - stages (tuple[bool], optional): Stages to apply plugin, length
            should be same as 'num_stages'.
            Default: None.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity. Default: True.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.

    Example:
        >>> from mmseg.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 multi_grid=None,
                 contract_dilation=False,
                 with_cp=False,
                 zero_init_residual=True,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')

        self.pretrained = pretrained
        self.zero_init_residual = zero_init_residual
        block_init_cfg = None
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
                block = self.arch_settings[depth][0]
                if self.zero_init_residual:
                    if block is BasicBlock:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm2'))
                    elif block is Bottleneck:
                        block_init_cfg = dict(
                            type='Constant',
                            val=0,
                            override=dict(name='norm3'))
        else:
            raise TypeError('pretrained must be a str or None')

        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
                init_cfg=block_init_cfg)
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.

        An example of plugins format could be :
        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:
            conv1-> conv2->conv3->yyy->zzz1->zzz2
        Suppose 'stage_idx=1', the structure of blocks in the stage would be:
            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        """Pack all blocks in a stage into a ``ResLayer``."""
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        """nn.Module: the normalization layer named "norm1" """
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        """Make stem layer for ResNet."""
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Identity()

    def _freeze_stages(self):
        """Freeze stages param and norm stats."""
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        """Forward function."""
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@MODELS.register_module()
class ResNetV1cnp(ResNetnoPool):
    """ResNetV1c variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1c replaces the 7x7 conv in
    the input stem with three 3x3 convs. For more details please refer to `Bag
    of Tricks for Image Classification with Convolutional Neural Networks
    <https://arxiv.org/abs/1812.01187>`_.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=False, **kwargs)


@MODELS.register_module()
class ResNetV1dnp(ResNetnoPool):
    """ResNetV1d variant described in [1]_.

    Compared with default ResNet(ResNetV1b), ResNetV1d replaces the 7x7 conv in
    the input stem with three 3x3 convs. And in the downsampling block, a 2x2
    avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super().__init__(deep_stem=True, avg_down=True, **kwargs)


# 

@MODELS.register_module()
class FPNHeadTail(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, tail_ratio=2, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.tail_ratio = tail_ratio

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.tail = ConvModule(
                        self.channels,
                        self.channels * self.tail_ratio ** 2,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)
        
    def tail_upsample(self, x):
        x = self.tail(x)
        x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", 
                      p1=self.tail_ratio, p2=self.tail_ratio, c=self.channels)
        return x  


    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.tail_upsample(output)
        output = self.cls_seg(output)
        return output

@MODELS.register_module()
class FPNHeadTail2(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, tail_ratio=2, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.tail_ratio = tail_ratio

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels * self.tail_ratio ** 2,
                        self.channels * self.tail_ratio ** 2,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        
    def tail_upsample(self, x):
        # x = self.tail(x)
        x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", 
                      p1=self.tail_ratio, p2=self.tail_ratio, c=self.channels)
        return x  


    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.tail_upsample(output)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class FPNHeadTail3(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))
        
        self.tail = ConvModule(
                        self.channels,
                        self.channels * 4,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg)

        
    def tail_upsample(self, x):
        x = self.tail(x)
        x = rearrange(x, "b (c p1 p2) h w -> b c (h p1) (w p2)", 
                      p1=2, p2=2, c=self.channels)
        return x  


    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        output = self.tail_upsample(output)
        output = self.tail_upsample(output)
        output = self.cls_seg(output)
        return output


class UpSamplexn(nn.Module):
    def __init__(self, n=2) -> None:
        super().__init__()
        self.n = n
    
    def forward(self, x):
        x = rearrange(
            x, "b (c p1 p2) h w -> b c (h p1) (w p2)", 
            p1=self.n, p2=self.n)
        return x
    
class UpSamplexnBottle(nn.Module):
    def __init__(self, in_channels, channels=None, n=2) -> None:
        super().__init__()
        channels = channels or in_channels
        assert in_channels % n == 0, f"channels {channels} not match n {n}"
        self.bottleneck = nn.Sequential()
        self.bottleneck.append(nn.Conv2d(in_channels, channels * 2, 3, padding=1)) 
        self.bottleneck.append(nn.ReLU())
        self.bottleneck.append(nn.Conv2d(channels * 2, channels, 3, padding=1)) 
        self.bottleneck.append(UpSamplexn(n))

    
    def forward(self, x):
        x = self.bottleneck(x)
        return x

class Depthwise_Pointwise_Conv(nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size,
                stride = 1,
                dilation= 1,
                bias: bool = True,
                padding_mode: str = 'zeros',  # TODO: refine this type
                device=None,
                dtype=None,
                activation=nn.Hardswish
                 ) -> None:
        super().__init__()
        self.dc = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, 
            padding=kernel_size//2, dilation=dilation, groups=in_channels,
            bias=bias, padding_mode=padding_mode, device=device, dtype=dtype
            )
        self.act = activation()
        self.pc = nn.Conv2d(
            in_channels, out_channels, 1, 
            bias=bias, padding_mode=padding_mode, device=device, dtype=dtype
            )

    def forward(self, x):
        x = self.dc(x)
        x = self.act(x)
        x = self.pc(x)
        return x

@MODELS.register_module()
class FPNHeadTail4(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, tail_ratio=2, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.tail_ratio = tail_ratio

        self.scale_heads = nn.ModuleList()

        scale_head0 = []
        scale_head0.append(
            ConvModule(
                self.in_channels[-1],
                self.in_channels[-1] * 2,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg))
        
        scale_head0.append(UpSamplexn(n=2))

        self.scale_heads.append(nn.Sequential(*scale_head0))

        for i in range(3):
            scale_head = []
            scale_head.append(
                ConvModule(
                    self.in_channels[3 - i],
                    self.in_channels[3 - i],
                    3,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            scale_head.append(UpSamplexn(n=2))

            self.scale_heads.append(nn.Sequential(*scale_head))

        self.tail = UpSamplexnBottle(self.in_channels[0] // 2, self.channels * 4, 2)


    def forward(self, inputs):

        x = self._transform_inputs(inputs)


        output = self.scale_heads[0](x[-1])  # x2, 1024
        for i in range(1, 4):
            # logger.info(f"{x[3-i].shape}, {output.shape}")
            output = torch.cat([x[3 -i], output], 1) # dim * 2
            output = self.scale_heads[i](output) # hwx2 dim/4
        # x16 128
        output = self.tail(output)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class FPNHeadTail5(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, tail_ratio=2, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.tail_ratio = tail_ratio

        self.scale_heads = nn.ModuleList()

        scale_head0 = []
        scale_head0.append(
            Depthwise_Pointwise_Conv(
                self.in_channels[-1],
                self.in_channels[-1] * 2,
                3,
                padding=1,
            ))
        scale_head0.append(nn.BatchNorm2d(self.in_channels[-1] * 2))
        scale_head0.append(nn.Hardswish())
        
        scale_head0.append(UpSamplexn(n=2))

        self.scale_heads.append(nn.Sequential(*scale_head0))

        for i in range(3):
            scale_head = []
            scale_head.append(
                ConvModule(
                    self.in_channels[3 - i],
                    self.in_channels[3 - i],
                    3,
                    padding=1,
                    groups=self.in_channels[3 - i],
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))
            
            scale_head.append(UpSamplexn(n=2))

            self.scale_heads.append(nn.Sequential(*scale_head))

        self.tail = UpSamplexnBottle(self.in_channels[0] // 2, self.channels * 4, 2)


    def forward(self, inputs):

        x = self._transform_inputs(inputs)


        output = self.scale_heads[0](x[-1])  # x2, 1024
        for i in range(1, 4):
            # logger.info(f"{x[3-i].shape}, {output.shape}")
            output = torch.cat([x[3 -i], output], 1) # dim * 2
            output = self.scale_heads[i](output) # hwx2 dim/4
        # x16 128
        output = self.tail(output)
        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class FPNHead2(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, bottle_channels=None, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.bottle_channels = bottle_channels or self.in_channels

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

        self.tail = ConvModule(
            self.channels * len(self.in_channels),
            self.channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.deeplabhead = ConvModule(
            self.channels,
            self.bottle_channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        
        self.conv_seg = nn.Conv2d(self.bottle_channels, self.out_channels, kernel_size=1)

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        outputs = []
        outputs.append(self.scale_heads[0](x[0]))
        for i in range(1, len(self.feature_strides)):
            # non inplace
            outputs.append(resize(
                self.scale_heads[i](x[i]),
                size=outputs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners))
        output = torch.cat(outputs, 1)

        output = self.cls_seg(output)
        return output
    
    def cls_seg(self, feat):
        """Classify each pixel."""
        feat = self.tail(feat)
        if self.dropout is not None:
            feat = self.dropout(feat)
        # DeepLabv3 Head
        feat = self.deeplabhead(feat)
        output = self.conv_seg(feat)
        return output



@MODELS.register_module()
class FPNHeadNeck(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, return_neck=False, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.return_neck = return_neck

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))


        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.global_neck = nn.Sequential(
            nn.Linear(self.in_channels[-1], self.channels * 2, bias=False),
            nn.BatchNorm1d(self.channels * 2),
            nn.ReLU(),
            nn.Linear(self.channels * 2, self.channels)
        )

    def feature_global(self, x):
        B, C, *_ = x.shape
        x = self.global_pool(x)
        x = x.view(B, C)
        x = self.global_neck(x)
        x = x.view(B, -1, 1, 1)
        return x


    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        # from loguru import logger 
        # logger.info(f"{x[0].shape, x[-1].shape}")
        # if self.return_neck:
        #     return output, self.feature_global(x[-1])
        # else:
        #     output = output + \
        #             self.feature_global(x[-1])

        #     output = self.cls_seg(output)
        #     return output
        
        output = output + \
                self.feature_global(x[-1])

        output = self.cls_seg(output)
        return output



# @MODELS.register_module()
# class FPNHead(BaseDecodeHead):
#     """Panoptic Feature Pyramid Networks.

#     This head is the implementation of `Semantic FPN
#     <https://arxiv.org/abs/1901.02446>`_.

#     Args:
#         feature_strides (tuple[int]): The strides for input feature maps.
#             stack_lateral. All strides suppose to be power of 2. The first
#             one is of largest resolution.
#     """

#     def __init__(self, feature_strides, **kwargs):
#         super().__init__(input_transform='multiple_select', **kwargs)
#         assert len(feature_strides) == len(self.in_channels)
#         assert min(feature_strides) == feature_strides[0]
#         self.feature_strides = feature_strides

#         self.scale_heads = nn.ModuleList()
#         for i in range(len(feature_strides)):
#             head_length = max(
#                 1,
#                 int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
#             scale_head = []
#             for k in range(head_length):
#                 scale_head.append(
#                     ConvModule(
#                         self.in_channels[i] if k == 0 else self.channels,
#                         self.channels,
#                         3,
#                         padding=1,
#                         conv_cfg=self.conv_cfg,
#                         norm_cfg=self.norm_cfg,
#                         act_cfg=self.act_cfg))
#                 if feature_strides[i] != feature_strides[0]:
#                     scale_head.append(
#                         Upsample(
#                             scale_factor=2,
#                             mode='bilinear',
#                             align_corners=self.align_corners))
#             self.scale_heads.append(nn.Sequential(*scale_head))

#     def forward(self, inputs):

#         x = self._transform_inputs(inputs)

#         output = self.scale_heads[0](x[0])
#         for i in range(1, len(self.feature_strides)):
#             # non inplace
#             output = output + resize(
#                 self.scale_heads[i](x[i]),
#                 size=output.shape[2:],
#                 mode='bilinear',
#                 align_corners=self.align_corners)

#         output = self.cls_seg(output)
#         return output


class Decoder_Block(nn.Module):
    # UNet Decoder block
    def __init__(self, x_channel, y_channel, dims, stride=2,align_corners=None,*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.align_corners = align_corners

        # use transpose conv to upsample
        if stride == 1:
            self.upconv = nn.Identity()
        else:
            self.upconv = nn.ConvTranspose2d(x_channel, x_channel, stride, stride)

        # Convolutional block: conv + bn + relu + conv + bn + relu
        self.conv = nn.Sequential(
            nn.Conv2d(x_channel + y_channel, dims, 3, padding=1),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(dims, dims, 3, padding=1),
            nn.BatchNorm2d(dims),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x = self.upconv(x)
        x = resize(
                x,
                size=y.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        x = torch.cat([x, y], dim=1)
        x = self.conv(x)
        return x


@MODELS.register_module()
class UNetHead(BaseDecodeHead):


    def __init__(self, feature_strides, ratio=1, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        # build decoder blocks: use Decoder_Block
        decoder_blocks = []
        last_channels = self.in_channels[-1]
        for i in range(len(feature_strides) - 2, -1, -1):
            next_channels = int(self.in_channels[i] * ratio)
            decoder_blocks.append(
                Decoder_Block(
                    last_channels,
                    self.in_channels[i],
                    next_channels,
                    stride=feature_strides[i + 1] // feature_strides[i],
                    align_corners=self.align_corners
                )
            )
            last_channels = next_channels
        self.decoder_blocks = nn.ModuleList(decoder_blocks)

        


    def forward(self, inputs):

        x = self._transform_inputs(inputs)[::-1]

        for i in range(len(self.feature_strides) - 1):
            if i == 0:
                output = self.decoder_blocks[i](x[i], x[i + 1])
            else:
                output = self.decoder_blocks[i](output, x[i + 1])

        output = self.cls_seg(output)
        return output


@MODELS.register_module()
class PointHeadNeck(BaseCascadeDecodeHead):
    """A mask point head use in PointRend.

    This head is implemented of `PointRend: Image Segmentation as
    Rendering <https://arxiv.org/abs/1912.08193>`_.
    ``PointHead`` use shared multi-layer perceptron (equivalent to
    nn.Conv1d) to predict the logit of input points. The fine-grained feature
    and coarse feature will be concatenate together for predication.

    Args:
        num_fcs (int): Number of fc layers in the head. Default: 3.
        in_channels (int): Number of input channels. Default: 256.
        fc_channels (int): Number of fc channels. Default: 256.
        num_classes (int): Number of classes for logits. Default: 80.
        class_agnostic (bool): Whether use class agnostic classification.
            If so, the output channels of logits will be 1. Default: False.
        coarse_pred_each_layer (bool): Whether concatenate coarse feature with
            the output of each fc layer. Default: True.
        conv_cfg (dict|None): Dictionary to construct and config conv layer.
            Default: dict(type='Conv1d'))
        norm_cfg (dict|None): Dictionary to construct and config norm layer.
            Default: None.
        loss_point (dict): Dictionary to construct and config loss layer of
            point head. Default: dict(type='CrossEntropyLoss', use_mask=True,
            loss_weight=1.0).
    """

    def __init__(self,
                 num_fcs=3,
                 coarse_pred_each_layer=True,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU', inplace=False),
                 **kwargs):
        super().__init__(
            input_transform='multiple_select',
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=dict(
                type='Normal', std=0.01, override=dict(name='fc_seg')),
            **kwargs)
        if point_sample is None:
            raise RuntimeError('Please install mmcv-full for '
                               'point_sample ops')

        self.num_fcs = num_fcs
        self.coarse_pred_each_layer = coarse_pred_each_layer

        fc_in_channels = sum(self.in_channels) + self.num_classes
        fc_channels = self.channels
        self.fcs = nn.ModuleList()
        for k in range(num_fcs):
            fc = ConvModule(
                fc_in_channels,
                fc_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
            self.fcs.append(fc)
            fc_in_channels = fc_channels
            fc_in_channels += self.num_classes if self.coarse_pred_each_layer \
                else 0
        self.fc_seg = nn.Conv1d(
            fc_in_channels,
            self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0)
        if self.dropout_ratio > 0:
            self.dropout = nn.Dropout(self.dropout_ratio)
        delattr(self, 'conv_seg')

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        
        self.global_neck = nn.Sequential(
            nn.Linear(self.in_channels[-1], self.channels * 2, bias=False),
            nn.BatchNorm1d(self.channels * 2),
            nn.ReLU(),
            nn.Linear(self.channels * 2, self.channels)
        )

    def feature_global(self, x):
        B, C, *_ = x.shape
        x = self.global_pool(x)
        x = x.view(B, C)
        x = self.global_neck(x)
        x = x.view(B, -1, 1, 1)
        return x

    def cls_seg(self, feat):
        """Classify each pixel with fc."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.fc_seg(feat)
        return output

    def forward(self, fine_grained_point_feats, coarse_point_feats):
        x = torch.cat([fine_grained_point_feats, coarse_point_feats], dim=1)
        for fc in self.fcs:
            x = fc(x)
            if self.coarse_pred_each_layer:
                x = torch.cat((x, coarse_point_feats), dim=1)
        return self.cls_seg(x)

    def _get_fine_grained_point_feats(self, x, points):
        """Sample from fine grained features.

        Args:
            x (list[Tensor]): Feature pyramid from by neck or backbone.
            points (Tensor): Point coordinates, shape (batch_size,
                num_points, 2).

        Returns:
            fine_grained_feats (Tensor): Sampled fine grained feature,
                shape (batch_size, sum(channels of x), num_points).
        """

        fine_grained_feats_list = [
            point_sample(_, points, align_corners=self.align_corners)
            for _ in x
        ]
        if len(fine_grained_feats_list) > 1:
            fine_grained_feats = torch.cat(fine_grained_feats_list, dim=1)
        else:
            fine_grained_feats = fine_grained_feats_list[0]

        return fine_grained_feats

    def _get_coarse_point_feats(self, prev_output, points):
        """Sample from fine grained features.

        Args:
            prev_output (list[Tensor]): Prediction of previous decode head.
            points (Tensor): Point coordinates, shape (batch_size,
                num_points, 2).

        Returns:
            coarse_feats (Tensor): Sampled coarse feature, shape (batch_size,
                num_classes, num_points).
        """

        coarse_feats = point_sample(
            prev_output, points, align_corners=self.align_corners)

        return coarse_feats

    def loss(self, inputs, prev_output, batch_data_samples: SampleList,
             train_cfg, **kwargs):
        """Forward function for training.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self._transform_inputs(inputs)
        with torch.no_grad():
            points = self.get_points_train(
                prev_output, calculate_uncertainty, cfg=train_cfg)
        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, points)
        coarse_point_feats = self._get_coarse_point_feats(prev_output, points)
        point_logits = self.forward(fine_grained_point_feats,
                                    coarse_point_feats)

        losses = self.loss_by_feat(point_logits, points, batch_data_samples)

        return losses

    def predict(self, inputs, prev_output, batch_img_metas: List[dict],
                test_cfg, **kwargs):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """

        x = self._transform_inputs(inputs)
        refined_seg_logits = prev_output.clone()
        for _ in range(test_cfg.subdivision_steps):
            refined_seg_logits = resize(
                refined_seg_logits,
                scale_factor=test_cfg.scale_factor,
                mode='bilinear',
                align_corners=self.align_corners)
            batch_size, channels, height, width = refined_seg_logits.shape
            point_indices, points = self.get_points_test(
                refined_seg_logits, calculate_uncertainty, cfg=test_cfg)
            fine_grained_point_feats = self._get_fine_grained_point_feats(
                x, points)
            coarse_point_feats = self._get_coarse_point_feats(
                prev_output, points)
            point_logits = self.forward(fine_grained_point_feats,
                                        coarse_point_feats)

            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_seg_logits = refined_seg_logits.reshape(
                batch_size, channels, height * width)
            refined_seg_logits = refined_seg_logits.scatter_(
                2, point_indices, point_logits)
            refined_seg_logits = refined_seg_logits.view(
                batch_size, channels, height, width)

        return self.predict_by_feat(refined_seg_logits, batch_img_metas,
                                    **kwargs)

    def loss_by_feat(self, point_logits, points, batch_data_samples, **kwargs):
        """Compute segmentation loss."""
        gt_semantic_seg = self._stack_batch_gt(batch_data_samples)
        point_label = point_sample(
            gt_semantic_seg.float(),
            points,
            mode='nearest',
            align_corners=self.align_corners)
        point_label = point_label.squeeze(1).long()

        loss = dict()
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        for loss_module in losses_decode:
            loss['point' + loss_module.loss_name] = loss_module(
                point_logits, point_label, ignore_index=self.ignore_index)

        loss['acc_point'] = accuracy(
            point_logits, point_label, ignore_index=self.ignore_index)
        return loss

    def get_points_train(self, seg_logits, uncertainty_func, cfg):
        """Sample points for training.

        Sample points in [0, 1] x [0, 1] coordinate space based on their
        uncertainty. The uncertainties are calculated for each point using
        'uncertainty_func' function that takes point's logit prediction as
        input.

        Args:
            seg_logits (Tensor): Semantic segmentation logits, shape (
                batch_size, num_classes, height, width).
            uncertainty_func (func): uncertainty calculation function.
            cfg (dict): Training config of point head.

        Returns:
            point_coords (Tensor): A tensor of shape (batch_size, num_points,
                2) that contains the coordinates of ``num_points`` sampled
                points.
        """
        num_points = cfg.num_points
        oversample_ratio = cfg.oversample_ratio
        importance_sample_ratio = cfg.importance_sample_ratio
        assert oversample_ratio >= 1
        assert 0 <= importance_sample_ratio <= 1
        batch_size = seg_logits.shape[0]
        num_sampled = int(num_points * oversample_ratio)
        point_coords = torch.rand(
            batch_size, num_sampled, 2, device=seg_logits.device)
        point_logits = point_sample(seg_logits, point_coords)
        # It is crucial to calculate uncertainty based on the sampled
        # prediction value for the points. Calculating uncertainties of the
        # coarse predictions first and sampling them for points leads to
        # incorrect results.  To illustrate this: assume uncertainty func(
        # logits)=-abs(logits), a sampled point between two coarse
        # predictions with -1 and 1 logits has 0 logits, and therefore 0
        # uncertainty value. However, if we calculate uncertainties for the
        # coarse predictions first, both will have -1 uncertainty,
        # and sampled point will get -1 uncertainty.
        point_uncertainties = uncertainty_func(point_logits)
        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points
        idx = torch.topk(
            point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
        shift = num_sampled * torch.arange(
            batch_size, dtype=torch.long, device=seg_logits.device)
        idx += shift[:, None]
        point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
            batch_size, num_uncertain_points, 2)
        if num_random_points > 0:
            rand_point_coords = torch.rand(
                batch_size, num_random_points, 2, device=seg_logits.device)
            point_coords = torch.cat((point_coords, rand_point_coords), dim=1)
        return point_coords

    def get_points_test(self, seg_logits, uncertainty_func, cfg):
        """Sample points for testing.

        Find ``num_points`` most uncertain points from ``uncertainty_map``.

        Args:
            seg_logits (Tensor): A tensor of shape (batch_size, num_classes,
                height, width) for class-specific or class-agnostic prediction.
            uncertainty_func (func): uncertainty calculation function.
            cfg (dict): Testing config of point head.

        Returns:
            point_indices (Tensor): A tensor of shape (batch_size, num_points)
                that contains indices from [0, height x width) of the most
                uncertain points.
            point_coords (Tensor): A tensor of shape (batch_size, num_points,
                2) that contains [0, 1] x [0, 1] normalized coordinates of the
                most uncertain points from the ``height x width`` grid .
        """

        num_points = cfg.subdivision_num_points
        uncertainty_map = uncertainty_func(seg_logits)
        batch_size, _, height, width = uncertainty_map.shape
        h_step = 1.0 / height
        w_step = 1.0 / width

        uncertainty_map = uncertainty_map.view(batch_size, height * width)
        num_points = min(height * width, num_points)
        point_indices = uncertainty_map.topk(num_points, dim=1)[1]
        point_coords = torch.zeros(
            batch_size,
            num_points,
            2,
            dtype=torch.float,
            device=seg_logits.device)
        point_coords[:, :, 0] = w_step / 2.0 + (point_indices %
                                                width).float() * w_step
        point_coords[:, :, 1] = h_step / 2.0 + (point_indices //
                                                width).float() * h_step
        return point_indices, point_coords
