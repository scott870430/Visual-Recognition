# visual recognition hw3
import os
import csv
import math
import shutil
import numpy as np
import logging

import torch
import torchvision.models
import transforms as T

import torchvision
import torchvision.models.detection as detection
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_instance_segmentation_model(num_classes,
                                    tl=3,
                                    hidden_layer=256):
    model = detection.maskrcnn_resnet50_fpn(pretrained=False,
                                            pretrained_backbone=True,
                                            trainable_backbone_layers=tl)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = hidden_layer

    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(istrain, isnorm=False):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if istrain:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))

    if isnorm:
        transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


class Logger(object):
    def __init__(self, log_file_name, logger_name, log_level=logging.DEBUG):
        # create a logger
        self.__logger = logging.getLogger(logger_name)

        # set the log level
        self.__logger.setLevel(log_level)

        # create a handler to write log file
        file_handler = logging.FileHandler(log_file_name)

        # create a handler to print on console
        console_handler = logging.StreamHandler()

        # define the output format of handlers
        formatter = logging.Formatter('[%(asctime)s] - ' +
                                      '[%(filename)s ' +
                                      'file line:%(lineno)d] - ' +
                                      '%(levelname)s: %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # add handler to logger
        self.__logger.addHandler(file_handler)
        self.__logger.addHandler(console_handler)

    def get_log(self):
        return self.__logger


def mkExpDir(args):
    if (os.path.exists(args.save_dir)):
        if (not args.reset):
            raise SystemExit('Error: save_dir "' +
                             args.save_dir +
                             '" already exists! ' +
                             'Please set --reset True to delete the folder.')
        else:
            shutil.rmtree(args.save_dir)

    os.makedirs(args.save_dir)
    # os.makedirs(os.path.join(args.save_dir, 'img'))
    print(os.path.join(args.save_dir, 'config.yaml'))
    shutil.copyfile('./config.yaml',
                    os.path.join(args.save_dir, 'config.yaml'))
    # if (not args.test):
    os.makedirs(os.path.join(args.save_dir, 'model'))

    # if args.test:
    #    os.makedirs(os.path.join(args.save_dir, 'save_results'))

    args_file = open(os.path.join(args.save_dir, 'args.txt'), 'w')
    for k, v in vars(args).items():
        args_file.write(k.rjust(30, ' ') + '\t' + str(v) + '\n')

    _logger = Logger(log_file_name=os.path.join(args.save_dir,
                                                args.log_file_name),
                     logger_name=args.logger_name).get_log()

    return _logger
