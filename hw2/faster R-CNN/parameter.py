# visual recognition hw2
import os
import csv
import torch
import torchvision.models
import transforms as T
import torchvision
import torchvision.models.detection as detection
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator


def get_model(classes=11):
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features

    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),),
                                       aspect_ratios=((0.5, 1.0, 1.5),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
                                            featmap_names=['0', '1', '2', '3'],
                                            output_size=7,
                                            sampling_ratio=2)

    # put the pieces together inside a FasterRCNN model
    model = FasterRCNN(backbone,
                       num_classes=classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    return model


def get_object_detection_model(num_classes, ispretrain=True):
    # load an object detection model pre-trained on COCO
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # replace the classifier with a new one
    num_classes = num_classes  # 11 class digit + background
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = detection.faster_rcnn.FastRCNNPredictor(
                                                    in_features, num_classes)

    return model


def get_transform(istrain):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if istrain:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    return T.Compose(transforms)

istest = False
version = 'version7'
Batch_size = 5
lr = 0.005
epochs = 20
