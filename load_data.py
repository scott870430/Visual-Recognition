import csv
import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data as Data
from torchvision import transforms
import parameter


class car_dataset(Data.Dataset):
    def __init__(self, istrain=True, transform=None):

        self.istrain = istrain
        self.transform = transform
        self.root = './dataset/'
        if self.istrain:
            # read training data
            self.train_data = []
            self.train_labels = []
            self.train_file = 'training_data/'
            with open(self.root + 'training_labels.csv',
                      newline='') as csvfile:
                rows = list(csv.reader(csvfile))
                for row in rows[1:]:
                    self.train_data.append(self.root + self.train_file +
                                           str(row[0]) + '.jpg')
                    self.train_labels.append(parameter.label2index[row[1]])
            if parameter.isTest:
                self.train_data = self.train_data[:10]
                self.train_labels = self.train_labels[:10]

            print("load", len(self.train_labels), "training data")
        else:
            self.test_data = []
            self.test_img_indexs = []
            self.test_file = 'testing_data/'
            import os
            testing_path = os.listdir(self.root + self.test_file)

            for file in testing_path:
                self.test_data.append(self.root + self.test_file + file)
                self.test_img_indexs.append(file[:-4])

    def __getitem__(self, index):
        if self.istrain:
            # training data
            img_path = self.train_data[index]
            label = self.train_labels[index]
        else:
            # testing data
            img_path = self.test_data[index]
            img_index = self.test_img_indexs[index]
        img = cv2.imread(img_path)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.istrain:
            return img, torch.from_numpy(np.array(label)).long()
        else:
            return img, img_index

    def __len__(self):

        if self.istrain:
            return len(self.train_data)
        else:
            return len(self.test_data)

if __name__ == '__main__':

    test = car_dataset(False, transform)
    print(test[10])
