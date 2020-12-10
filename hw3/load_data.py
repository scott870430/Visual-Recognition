import function

import os
import cv2
import numpy as np
import json
import glob

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image

import torch
import torch.utils.data as Data
import torchvision
from coco_utils import convert_coco_poly_to_mask


class TPVDataset(object):
    def __init__(self, mode, transforms=None, isTEST=False):
        self.mode = mode
        self.transforms = transforms

        if self.mode == 'train':
            self.anno_path = os.path.join('dataset', 'pascal_train.json')
        elif self.mode == 'test':
            self.anno_path = os.path.join('dataset', 'test.json')

        self.img_root = os.path.join('dataset', self.mode + '_images')
        self.coco = COCO(self.anno_path)
        self.ids = list(self.coco.imgs.keys())
        if isTEST:
            self.ids = self.ids[:30]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(ids=img_id)
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annids)

        img_path = os.path.join(self.img_root, img_info[0]['file_name'])
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.open(img_path)
        # get bounding box coordinates for each mask

        boxes = []
        masks = []
        labels = []
        iscrowd = []
        areas = []
        for i in range(len(annids)):
            tmp = anns[i]['bbox']
            bbox = [tmp[0], tmp[1], tmp[0] + tmp[2], tmp[1] + tmp[3]]
            boxes.append(bbox)
            # print(anns[i]['bbox'])
            masks.append(self.coco.annToMask(anns[i]))
            labels.append(anns[i]['category_id'])
            iscrowd.append(anns[i]['iscrowd'])
            areas.append(anns[i]['area'])
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks)
        image_id = torch.tensor([img_id], dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # suppose all instances are not crowd
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ids)


class TPVDataset_test(object):
    def __init__(self, transform, isTEST=False):
        self.transforms = transform

        self.img_root = os.path.join('.', 'dataset', 'test_images')
        self.coco = COCO(os.path.join('.', 'dataset', 'test.json'))
        self.ids = list(self.coco.imgs.keys())
        if isTEST:
            self.ids = self.ids[:30]

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(ids=img_id)
        annids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(annids)

        img_path = os.path.join(self.img_root, img_info[0]['file_name'])
        img = Image.open(img_path)
        # get bounding box coordinates for each mask

        if self.transforms is not None:
            img = self.transforms(img)
        return img, img_id

    def __len__(self):
        return len(self.ids)

if __name__ == '__main__':
    coco = COCO('./dataset/pascal_train.json')
    dataset = TPVDataset_test(None)
    print(dataset.ids)
    dataset = TPVDataset('train', None)
    print(dataset.ids)
    for i in range(1349):
        img, obj_annos = dataset[i]
