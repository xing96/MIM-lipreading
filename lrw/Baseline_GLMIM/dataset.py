# encoding: utf-8
import numpy as np
import glob
import time
import cv2



def load_file(filename):
    arrays = np.load(filename)
    # arrays = np.stack([cv2.cvtColor(arrays[_], cv2.COLOR_BGR2GRAY)
    #                   for _ in range(29)], axis=0)
    arrays = arrays / 255.
    return arrays


class MyDataset():
    def __init__(self, folds, path):
        self.folds = folds
        self.path = path
        with open('label_sorted.txt') as myfile:
            lines = myfile.read().splitlines()
        self.data_dir = [self.path + item for item in lines]
        self.data_files = glob.glob(self.path+'*/'+self.folds+'/*.npy')
        self.labels = {}
        for idx, label in enumerate(list(open('label_sorted.txt'))):
            label = label.strip()
            self.labels[label] = idx

    def __getitem__(self, idx):
        inputs = load_file(self.data_files[idx])
        labels = self.labels[self.data_files[idx].split('/')[-1].split('_')[0]]
        return inputs, labels

    def __len__(self):
        return len(self.data_files)

