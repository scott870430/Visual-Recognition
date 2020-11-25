# The new config inherits a base config to highlight the necessary modification
_base_ = 'yolo/yolov3_d53_320_273e_coco.py'

# change the num_classes in head to match the dataset's annotation
input_size = 320
model = dict(
    bbox_head=dict(
        num_classes=10,))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('10', '1', '2', '3', '4', '5', '6', '7', '8', '9')

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'),
    val=dict(
        img_prefix='../dataset/train/',
        classes=classes,
        ann_file='../dataset/train/svhn_coco.json'),
    test=dict(
        img_prefix='../dataset/test/',
        classes=classes,
        ann_file='../dataset/test/svhn_coco_test.json'))


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
