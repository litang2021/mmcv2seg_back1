w = [0.2783, 0.9765, 1.0572, 1.8239, 0.5924, 1.2717]

custom_imports=dict(imports='customed', allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
dataset_type = 'ISIC2018'
data_root = 'data'
scale_size = (384, 576)
crop_size = (224, 224)
batch_size = 16


norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=(224, 224))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(224, 224)),
    pretrained=None,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
        )),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=6,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=
        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.),
        # dict(type='DiceLoss', loss_weight=1.,class_weight=[0.2, 1.2, 1.2, 1.2, 1.2, 1.])#]
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

######



train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=scale_size,
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[{
            'type': 'Resize',
            'scale_factor': 0.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 0.75,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.0,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.25,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.5,
            'keep_ratio': True
        }, {
            'type': 'Resize',
            'scale_factor': 1.75,
            'keep_ratio': True
        }],
                    [{
                        'type': 'RandomFlip',
                        'prob': 0.0,
                        'direction': 'horizontal'
                    }, {
                        'type': 'RandomFlip',
                        'prob': 1.0,
                        'direction': 'horizontal'
                    }], [{
                        'type': 'LoadAnnotations'
                    }], [{
                        'type': 'PackSegInputs'
                    }]])
]
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,

    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        balanced=False,
        data_prefix=dict(
            img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations'),
            dict(
                type='RandomResize',
                scale=scale_size,
                ratio_range=(0.5, 2.0),
                keep_ratio=True),
            dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='PackSegInputs')
        ]))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=crop_size, keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='img_dir/val', seg_map_path='ann_dir/val'),
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=crop_size, keep_ratio=True),
            dict(type='LoadAnnotations'),
            dict(type='PackSegInputs')
        ]))
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False
tta_model = dict(type='SegTTAModel')
optimizer = dict(type='SGD', lr=0.001 * batch_size / 10, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=None)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.0001,
        power=0.9,
        begin=0,
        end=80000,
        by_epoch=False)
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=int(5000 * 64 / batch_size), val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=500),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# load_from = "work_dirs/fpn_r50_b0w5/iter_5000.pth"



#####


# dataset_type = 'ADE20KDataset'
# data_root = 'data/ade/ADEChallengeData2016'
# crop_size = (512, 512)
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(
#         type='RandomResize',
#         scale=(2048, 512),
#         ratio_range=(0.5, 2.0),
#         keep_ratio=True),
#     dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='PhotoMetricDistortion'),
#     dict(type='PackSegInputs')
# ]
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#     dict(type='LoadAnnotations', reduce_zero_label=True),
#     dict(type='PackSegInputs')
# ]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[[{
#             'type': 'Resize',
#             'scale_factor': 0.5,
#             'keep_ratio': True
#         }, {
#             'type': 'Resize',
#             'scale_factor': 0.75,
#             'keep_ratio': True
#         }, {
#             'type': 'Resize',
#             'scale_factor': 1.0,
#             'keep_ratio': True
#         }, {
#             'type': 'Resize',
#             'scale_factor': 1.25,
#             'keep_ratio': True
#         }, {
#             'type': 'Resize',
#             'scale_factor': 1.5,
#             'keep_ratio': True
#         }, {
#             'type': 'Resize',
#             'scale_factor': 1.75,
#             'keep_ratio': True
#         }],
#                     [{
#                         'type': 'RandomFlip',
#                         'prob': 0.0,
#                         'direction': 'horizontal'
#                     }, {
#                         'type': 'RandomFlip',
#                         'prob': 1.0,
#                         'direction': 'horizontal'
#                     }], [{
#                         'type': 'LoadAnnotations'
#                     }], [{
#                         'type': 'PackSegInputs'
#                     }]])
# ]
# train_dataloader = dict(
#     batch_size=2,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type='ADE20KDataset',
#         data_root='data/ade/ADEChallengeData2016',
#         data_prefix=dict(
#             img_path='images/training', seg_map_path='annotations/training'),
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', reduce_zero_label=True),
#             dict(
#                 type='RandomResize',
#                 scale=(2048, 512),
#                 ratio_range=(0.5, 2.0),
#                 keep_ratio=True),
#             dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
#             dict(type='RandomFlip', prob=0.5),
#             dict(type='PhotoMetricDistortion'),
#             dict(type='PackSegInputs')
#         ]))
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='ADE20KDataset',
#         data_root='data/ade/ADEChallengeData2016',
#         data_prefix=dict(
#             img_path='images/validation',
#             seg_map_path='annotations/validation'),
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#             dict(type='LoadAnnotations', reduce_zero_label=True),
#             dict(type='PackSegInputs')
#         ]))
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type='ADE20KDataset',
#         data_root='data/ade/ADEChallengeData2016',
#         data_prefix=dict(
#             img_path='images/validation',
#             seg_map_path='annotations/validation'),
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='Resize', scale=(2048, 512), keep_ratio=True),
#             dict(type='LoadAnnotations', reduce_zero_label=True),
#             dict(type='PackSegInputs')
#         ]))
# val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
# default_scope = 'mmseg'
# env_cfg = dict(
#     cudnn_benchmark=True,
#     mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
#     dist_cfg=dict(backend='nccl'))
# vis_backends = [dict(type='LocalVisBackend')]
# visualizer = dict(
#     type='SegLocalVisualizer',
#     vis_backends=[dict(type='LocalVisBackend')],
#     name='visualizer')
# log_processor = dict(by_epoch=False)
# log_level = 'INFO'
# load_from = None
# resume = False
# tta_model = dict(type='SegTTAModel')
# optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
# optim_wrapper = dict(
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=6e-05, betas=(0.9, 0.999), weight_decay=0.01),
#     paramwise_cfg=dict(
#         custom_keys=dict(
#             absolute_pos_embed=dict(decay_mult=0.0),
#             relative_position_bias_table=dict(decay_mult=0.0),
#             norm=dict(decay_mult=0.0))))
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
#         end=1500),
#     dict(
#         type='PolyLR',
#         eta_min=0.0,
#         power=1.0,
#         begin=1500,
#         end=160000,
#         by_epoch=False)
# ]
# train_cfg = dict(
#     type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
# val_cfg = dict(type='ValLoop')
# test_cfg = dict(type='TestLoop')
# default_hooks = dict(
#     timer=dict(type='IterTimerHook'),
#     logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
#     param_scheduler=dict(type='ParamSchedulerHook'),
#     checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=16000),
#     sampler_seed=dict(type='DistSamplerSeedHook'),
#     visualization=dict(type='SegVisualizationHook'))
# checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'
