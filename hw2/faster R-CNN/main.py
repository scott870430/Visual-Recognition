import load_data
import parameter
from engine import train_one_epoch, evaluate


import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(parameter.version)
# draw tranining loss curve and test error rate curve


def draw_curve(lines, title, labels, needtosave=False):
    plt.figure()
    for line, label in zip(lines, labels):
        plt.plot([e for e in range(1, len(line) + 1)], line, label=label)

    plt.title(title)
    plt.legend()
    if needtosave:
        plt.savefig('./figure/' + str(parameter.version) + title + ".png")
    else:
        plt.show()

if __name__ == '__main__':

    dataset = load_data.digit_dataset('train',
                                      parameter.get_transform(istrain=True))
    val_dataset = load_data.digit_dataset(
                                    'train',
                                    parameter.get_transform(istrain=False))
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    val_dataset = torch.utils.data.Subset(val_dataset, indices[-50:])

    train_loader = Data.DataLoader(dataset,
                                   batch_size=parameter.Batch_size,
                                   shuffle=True, num_workers=4,
                                   collate_fn=torch.utils.collate_fn)

    val_loader = Data.DataLoader(val_dataset,
                                 batch_size=parameter.Batch_size,
                                 shuffle=False, num_workers=4,
                                 collate_fn=torch.utils.collate_fn)

    # model = parameter.get_object_detection_model(10, ispretrain = True)
    model = parameter.get_model(11)
    model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=parameter.lr,
                          momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                                                    optimizer, T_0=1, T_mult=2)

    train_acc = []
    training_loss_curve = []
    for epoch in range(1, parameter.epochs+1):
        print(f'epoch {epoch}')
        metric_logger = train_one_epoch(model, optimizer,
                                        train_loader, device,
                                        epoch, print_freq=3500)

        lr_scheduler.step()

        coco_evaluator = evaluate(model, val_loader, device=device)
        print(coco_evaluator.coco_eval['bbox'].stats[0])

        train_acc.append(coco_evaluator.coco_eval['bbox'].stats[0])
        training_loss_curve.append(metric_logger.meters['loss'].avg)
        model_path = './model'
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        torch.save(model.state_dict(),
                   model_path + '/model_' + parameter.version + str(epoch))

    draw_curve([train_acc], 'train_acc', ['acc'], needtosave=True)
    draw_curve([training_loss_curve],
               'training_loss',
               ['training_loss'],
               needtosave=True)

    print("END")
