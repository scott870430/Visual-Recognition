import load_data
import function
import utils
from engine import train_one_epoch, evaluate
from config import get_cfg_defaults

import os
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models


# draw tranining loss curve and test error rate curve
def draw_curve(lines, title, labels, version, needtosave=False):
    plt.figure()
    for line, label in zip(lines, labels):
        plt.plot([e for e in range(1, len(line) + 1)], line, label=label)

    plt.title(title)
    plt.legend()
    if needtosave:
        plt.savefig('./figure/' + str(version) + title + ".png")
    else:
        plt.show()


def main(args):
    print(args)
    _logger = function.mkExpDir(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    lr = cfg.TRAIN.LR
    version = args.version
    epochs = cfg.TRAIN.EPOCHS
    trainable = cfg.TRAIN.TRAINABLE
    Batch_size = cfg.TRAIN.BATCH_SIZE
    class_number = cfg.TRAIN.CLASS_NUMBER
    hidden_layer = cfg.TRAIN.HIDDEN_LAYER
    lr_step = cfg.TRAIN.LR_STEP
    lr_gamma = cfg.TRAIN.LR_GAMMA
    isnorm = cfg.TRAIN.ISNORM
    _logger.info(version)
    print(version)

    train_transform = function.get_transform(True, isnorm)
    test_transform = function.get_transform(False, isnorm)
    dataset = load_data.TPVDataset('train',
                                   transforms=train_transform,
                                   isTEST=args.test)
    val_dataset = load_data.TPVDataset('train',
                                       transforms=test_transform,
                                       isTEST=args.test)
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-20])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-20:])
    # print(len(dataset[0][1][1]['segmentation'][0]))
    train_loader = Data.DataLoader(dataset,
                                   batch_size=Batch_size,
                                   shuffle=True,
                                   num_workers=4,
                                   collate_fn=utils.collate_fn
                                   )
    val_loader = Data.DataLoader(val_dataset,
                                 batch_size=Batch_size,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=utils.collate_fn)
    model = function.get_instance_segmentation_model(class_number,
                                                     trainable,
                                                     hidden_layer
                                                     )
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params,
                          lr=lr,
                          momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=lr_step,
                                                   gamma=lr_gamma)

    train_acc = []
    training_loss_curve = []
    for epoch in range(1, epochs+1):
        print(f'epoch {epoch}')
        _logger.info('epoch: ' + str(epoch))
        metric_logger = train_one_epoch(model, optimizer,
                                        train_loader, device,
                                        epoch, print_freq=3500)
        _logger.info(metric_logger)
        lr_scheduler.step()
        coco_evaluator = evaluate(model, val_loader, device=device)
        print(coco_evaluator.coco_eval['bbox'].stats[0])
        print(coco_evaluator.coco_eval['segm'].stats[0])
        _logger.info("training segmentation acc: " +
                     str(coco_evaluator.coco_eval['segm'].stats[0]))
        train_acc.append(coco_evaluator.coco_eval['segm'].stats[0])
        training_loss_curve.append(metric_logger.meters['loss'].avg)
        torch.save(model.state_dict(),
                   os.path.join(args.save_dir, 'model', version + str(epoch)))

    draw_path = './figure'
    if not os.path.isdir(draw_path):
        os.makedirs(draw_path)
    draw_curve([train_acc], 'train_acc', ['acc'], version, needtosave=True)
    draw_curve([training_loss_curve],
               'training_loss',
               ['training_loss'],
               version,
               needtosave=True)

    print("END")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ids', '-t',
                        type=str,
                        default='7',
                        help='use which gpu')

    parser.add_argument('--config', '-c',
                        type=str,
                        default='config.yaml',
                        help='using config'
                        )
    parser.add_argument('--save-dir',
                        type=str,
                        # default='./train/test',
                        help='path for saveing log'
                        )
    parser.add_argument('--reset',
                        default=True,
                        type=bool,
                        help='reset log'
                        )
    parser.add_argument('--test',
                        default=False,
                        type=bool,
                        help='is test?'
                        )
    parser.add_argument('--log-file-name',
                        default="log",
                        type=str,
                        help='log file name'
                        )
    parser.add_argument('--logger-name',
                        default='train',
                        type=str,
                        help='logger name'
                        )
    parser.add_argument('--version',
                        type=str,
                        help='model version'
                        )
    args = parser.parse_args()
    main(args)
