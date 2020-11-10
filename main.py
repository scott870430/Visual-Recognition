import load_data
import parameter

import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models


os.environ["CUDA_VISIBLE_DEVICES"] = "6"
version = 'resnext50_32x4d_v3'
print(version)
# draw tranining loss curve and test error rate curve


def draw_curve(lines, title, labels, needtosave=False):
    plt.figure()
    for line, label in zip(lines, labels):
        plt.plot([e for e in range(1, len(line) + 1)], line, label=label)

    plt.title(title)
    plt.legend()
    if needtosave:
        plt.savefig('./figure/' + str(version) + title + ".png")
    else:
        plt.show()

if __name__ == '__main__':

	save_model_path = './model'
    train_transform = transforms.Compose([
        transforms.Resize(parameter.image_size),
        # random horizontal flip image
        transforms.RandomHorizontalFlip(p=0.5),
        # random rotate image
        transforms.RandomRotation(10),
        # change the data from 0~255 to 0~1 and the format to tensor
        transforms.ToTensor(),
        # normalize the data to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    # load training data
    train_loader = Data.DataLoader(
        load_data.car_dataset(True, train_transform),
        batch_size=parameter.Batch_size,
        shuffle=True,
        num_workers=4
        )
    # load model and modify classifier
    model = models.resnext50_32x4d(pretrained=True)
    fc_feature = model.fc.in_features
    model.fc = nn.Linear(fc_feature, 196)
    model = model.to(parameter.device)

    Loss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=parameter.lr,
                          momentum=0.9,
                          weight_decay=5e-4)

    train_acc, training_loss_curve = [], []

    for epoch in range(1, parameter.epochs+1):
        print(f'epoch {epoch}')
        # train
        model.train()
        total_number, train_correct, total_loss, running_loss = 0, 0, 0, 0

        for batch, (data, label) in enumerate(train_loader, 1):
            data, label = data.to(parameter.device), label.to(parameter.device)
            optimizer.zero_grad()
            # prediction
            output = model(data)
            # top1 prediction
            top1 = torch.max(output, 1)[1]
            # compute top1 correct
            train_correct += (top1 == label).sum().item()
            loss = Loss(output, label)  # compute loss
            loss.backward()  # compute gradient
            optimizer.step()  # update gradient

            running_loss += loss.item()
            total_loss += loss.item()
            if batch % 1000 == 0:
                print(f'batch {batch} loss: {running_loss / 100}')
                running_loss = 0

            total_number += len(data)
        print(f'training loss: {total_loss / batch}')
        print(f'training, top1 acc: {train_correct / total_number}')
        train_acc.append(train_correct / total_number)
        training_loss_curve.append(total_loss / total_number)

        # if epoch == parameter.epochs:
        if not os.path.isdir(save_model_path):
        	os.makedirs(save_model_path)
        torch.save(model.state_dict(), save_model_path + '/model_' + version)

    draw_curve([train_acc], 'training_acc', ['acc'], needtosave=True)
    draw_curve([training_loss_curve],
    	       'training_loss', ['training_loss'], needtosave=True)

    print("END")
