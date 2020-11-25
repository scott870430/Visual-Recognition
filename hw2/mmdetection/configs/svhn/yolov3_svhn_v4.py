# The new config inherits a base config to highlight the necessary modification
_base_ = 'yolo/yolov3_d53_mstrain-608_273e_coco.py'

# change the num_classes in head to match the dataset's annotation
model = dict(
    bbox_head=dict(
        num_classes=10,))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('10', '1', '2', '3', '4', '5', '6', '7', '8', '9')
img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(608, 608),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=10,
    workers_per_gpu=4,
    train=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',
        pipeline=train_pipeline),
    val=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='../dataset/test/',
        classes=classes,
        ann_file='../dataset/test/svhn_coco_test.json',
        pipeline=test_pipeline))


optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,  # same as burn-in in darknet
    warmup_ratio=0.1,
    step=[218, 246])
# runtime settings
total_epochs = 30
evaluation = dict(interval=1, metric=['bbox'])
