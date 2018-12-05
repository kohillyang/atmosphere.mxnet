import pandas as pd
import numpy as np
import os
from PIL import Image
import mxnet as mx
from sklearn.model_selection import train_test_split
import hashlib
import cv2


class ATDataset(object):
    def __init__(self, is_train = False):
        self.dataset_root="/media/kohill/data/kohill/dataset_qixiang/qixiang/train"
        csv_path = os.path.join(self.dataset_root, "train.csv")
        objs = np.array(pd.read_csv(csv_path))
        self.objs = {}
        self.labels = {}
        for fname, label in objs:
            f = os.path.join(self.dataset_root, "data", fname)
            md5 = hashlib.md5()
            md5.update(open(f, "rb").read())
            self.objs[md5.hexdigest()] = f
            self.labels[f] = label
        print(len(self.objs))
        names = list(self.objs.keys())
        names.sort()
        self.train_names, self.val_names = train_test_split(names, random_state=42, test_size=.1)
        if is_train:
            self.names = self.train_names
        else:
            self.names = self.val_names

        self._transforms = None

    def __len__(self):
        return len(self.names)

    def at_with_image_path(self, idx):
        path = self.objs[self.names[idx]]
        label = self.labels[path]
        return path, label-1

    def __getitem__(self, idx):
        path, label = self.at_with_image_path(idx)
        image = Image.open(path)
        image = np.array(image)
        if len(image.shape) == 2:
            image = image[:,:,np.newaxis]
            image = image[:,:,(0,0,0)]
        image = image[:, :, :3]
        if self._transforms is not None:
            image = self._transforms(image)
        return image, label

    def transform_first(self, transform):
        self._transforms = transform
        return self


if __name__ == '__main__':
    da = ATDataset()
    for image, label in da:
        print(label)