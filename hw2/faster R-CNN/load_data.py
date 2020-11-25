import parameter

import os
import csv
import cv2
import h5py
import numpy as np
import scipy.io as sio
from PIL import Image

import torch
import torch.utils.data as Data


def get_name(index, hdf5_data):
    name = hdf5_data['/digitStruct/name']
    return ''.join([chr(v[0]) for v in hdf5_data[name[index][0]][()]])


def get_bbox(index, hdf5_data):
    attrs = {}
    item = hdf5_data['digitStruct']['bbox'][index].item()
    for key in ['label', 'left', 'top', 'width', 'height']:
        attr = hdf5_data[item][key]
        '''
        values = [hdf5_data[attr[()][i].item()][()][0][0]
        for i in range(len(attr))] if len(attr) > 1 else [attr[()][0][0]]
        '''
        if len(attr) > 1:
        	values = [hdf5_data[attr[()][i].item()][()][0][0]
                      for i in range(len(attr))]
        else:
            [attr[()][0][0]]
        attrs[key] = values
    return attrs


class digit_dataset(Data.Dataset):
    def __init__(self, mode, transform=None):

        self.mode = mode
        self.transform = transform
        self.root = 'dataset'
        if self.mode == 'train':
            # read training data
            self.file = 'digitStruct.mat'
            self.path = os.path.join('.', self.root, self.mode)
            self.train_label_path = os.path.join(self.root,
                                                 self.mode,
                                                 self.file)
            self.mat = h5py.File(self.train_label_path, "r")

        elif self.mode == 'test':
            self.path = os.path.join('.', self.root, self.mode)
            self.test_image_path = []
            self.test_idx = []
            import re

            def convert(text):
                return int(text) if text.isdigit() else text.lower()

            def Sort(key):
                return [convert(c) for c in re.split('([0-9]+)', key)]

            def sorted_alphanumeric(data):
                return sorted(data, key=Sort)

            sorted_file = sorted_alphanumeric(os.listdir(self.path))
            for file in sorted_file:
                if file[-4:] != ".png":
                    continue
                self.test_image_path.append(os.path.join(self.path, file))
                self.test_idx.append(file)

    def __getitem__(self, index):
        if self.mode == 'train':
            # training data
            img_path = get_name(index, self.mat)
            img_path = os.path.join(self.path, img_path)
            temp = get_bbox(index, self.mat)
            label = temp['label']
            left = temp['left']
            top = temp['top']
            width = temp['width']
            height = temp['height']

        elif self.mode == 'test':
            # testing data
            img_path, idx = self.test_image_path[index], self.test_idx[index]

        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        if self.mode == 'test':
            img, _ = self.transform(img, None)
            return img, idx

        boxes = []
        right = np.add(left, width)
        down = np.add(top, height)
        for e in range(len(label)):
            box = [left[e], top[e], right[e], down[e]]
            boxes.append(box)

        iscrowd = torch.zeros((len(label)), dtype=torch.float32)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        label = torch.as_tensor(label, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = label
        target["image_id"] = torch.tensor([index])
        target["area"] = (
                         boxes[:, 3] - boxes[:, 1]) * (
                         boxes[:, 2] - boxes[:, 0])
        target["iscrowd"] = iscrowd
        if self.transform is not None:
            img, target = self.transform(img, target)

        if self.mode == 'train':
            return img, target

    def __len__(self):

        if parameter.istest:
            return 100
        if self.mode == 'train':
            return len(self.mat['/digitStruct/name'])
        elif self.mode == 'test':
            return len(self.test_image_path)


if __name__ == '__main__':

    print(os.path.join('dataset', 'train', 'digitStruct.mat'))
    mat = h5py.File(os.path.join('dataset', 'train', 'digitStruct.mat'), "r")
    print(get_bbox(24, mat))
    A = get_bbox(24, mat)
    print(np.add(A['top'], A['height']))
