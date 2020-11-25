# The new config inherits a base config to highlight the necessary modification
_base_ = [
    'ssd/ssd300_coco.py'
]
# datas
# We also need to change the num_classes in head to match the dataset's annotation
input_size = 300
model = dict(
    backbone=dict(input_size=input_size),
    bbox_head=dict(
        num_classes=10,))

# Modify dataset related settings
dataset_type = 'CocoDataset'
classes = ('10', '1', '2', '3', '4', '5', '6', '7', '8', '9')
'''
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',
        dataset=dict(
        pipline=[
        dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
        ])),
    val=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'),
    test=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'))
'''
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,                                                 # 数据集类型
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',                               # 数据集的图片路径
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            #dict(type='Pad', size_divisor=32),
            #dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),  
    val=dict(
        type=dataset_type,                                                 # 数据集类型
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',                               # 数据集的图片路径
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            #dict(type='Pad', size_divisor=32),
            #dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),  
    test=dict(
    type=dataset_type,                                                 # 数据集类型
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',                               # 数据集的图片路径
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=False),
            dict(type='Resize', img_scale=(300, 300), keep_ratio=False),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            #dict(type='Pad', size_divisor=32),
            #dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels'])
        ]))
# We can use the pre-trained Mask RCNN model to obtain higher performance
#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'