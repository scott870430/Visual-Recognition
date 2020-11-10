import load_data
import parameter

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models as models
from torchvision import transforms
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "9"
version = 'resnext50_32x4d_v6'
if __name__ == '__main__':
    Batch_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnext50_32x4d(pretrained=True)
    fc_feature = model.fc.in_features
    model.fc = nn.Linear(fc_feature, 196)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./model/model_' + version + '19',
                          map_location=lambda storage, loc: storage))
    model.to(device)
    test_transform = transforms.Compose([
        transforms.Resize(parameter.image_size),
        # change the data from 0~255 to 0~1 and the format to tensor
        transforms.ToTensor(),
        # normalize the data to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    test_loader = Data.DataLoader(load_data.car_dataset(False, test_transform),
                                  batch_size=Batch_size, shuffle=False, num_workers=4)
    
    prediction_path = './prediction'
    if not os.path.isdir(prediction_path):
        	os.makedirs(prediction_path)
    # generate prediction result
    with torch.no_grad():
        # test
        with open(prediction_path + '/' + version + '.txt', 'w') as f:
            print("id,label", file=f)
            for batch, (data, img_index) in enumerate(test_loader, 1):
                data = data.to(device)
                output = model(data)
                top1 = torch.max(output, 1)[1]
                for e, i in zip(top1, img_index):
                    print(i, ",", parameter.index2label[e.item()],
                    	  file=f, sep='')

    print("END")
