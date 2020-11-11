import os
import csv
import torch

isTest = False
image_size = (700, 700)
Batch_size = 8
lr = 0.001
epochs = 20
# os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label2index = {}
index2label = {}
class_number = 0
# match label to index
with open('./dataset/training_labels.csv', newline='') as csvfile:
    rows = list(csv.reader(csvfile))
    for row in rows[1:]:
        if row[1] not in label2index:
            label2index[row[1]] = class_number
            index2label[class_number] = row[1]
            class_number += 1
