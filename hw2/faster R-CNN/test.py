import load_data
import parameter
import argparse

import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
import torchvision.models as models


os.environ["CUDA_VISIBLE_DEVICES"] = "7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', '-v',
                        type=str,
                        default='version76',
                        help='version of model')
    args = parser.parse_args()

    Batch_size = 8
    # '''
    version = args.version
    print(version)
    # model = parameter.get_object_detection_model(11)
    model = parameter.get_model(11)
    model.load_state_dict(torch.load('./model/model_' + version,
                          map_location=lambda storage, loc: storage))
    model.to(device)

    # '''
    print("test dataset!")
    test_transform = parameter.get_transform(istrain=False)
    test_dataset = load_data.digit_dataset('test',
                                           test_transform)
    test_loader = Data.DataLoader(test_dataset,
                                  batch_size=Batch_size,
                                  shuffle=False,
                                  num_workers=4,
                                  collate_fn=torch.utils.collate_fn)

    # model = parameter.get_object_detection_model(2)
    # model.load_state_dict(torch.load('./model/model_' + parameter.version,\
    #     map_location=lambda storage, loc: storage))
    prediction = []
    with torch.no_grad():
        model.eval()
        for batch, (images, idx) in enumerate(test_loader, 1):
            print(idx)
            images = list(img.to(device) for img in images)
            output = model(images)
            for i in range(len(output)):
                image_path = './dataset/test/' + idx[i]
                img = cv2.imread(image_path)
                y, x, _ = img.shape
                d = {}
                d["bbox"], d["score"], d["label"] = [], [], []
                for box, label, score in zip(output[i]['boxes'],
                                             output[i]['labels'],
                                             output[i]['scores']):
                    box, score, label = box.cpu(), score.cpu(), label.cpu()
                    box = [round(e.item()) for e in box]
                    cv2.rectangle(img,
                                  (box[0], box[1]),
                                  (box[2], box[3]),
                                  (0, 0, 255), 2)
                    d["label"].append(label.item())
                    d["bbox"].append([box[1], box[0], box[3], box[2]])
                    d["score"].append(score.item())

                cv2.imwrite("./figure/test/" + idx[i], img)
                prediction.append(d)

    import json
    with open('./prediction/' + version + '0856165.json', "w") as f:
        json.dump(prediction, f)

    print("END")
