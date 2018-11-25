# filelist.py

import math
import torch
import numpy as np
import utils as utils
import torch.utils.data as data
from sklearn.utils import shuffle
import datasets.loaders as loaders
import torchvision.transforms as transforms

class FileList(data.Dataset):
    def __init__(self, ifile, lfile=None, split_train=1.0, split_test=0.0, train=True, 
        transform_train=None, transform_test=None, loader_input=loaders.loader_image, loader_label=loaders.loader_torch):
            
        self.ifile = ifile
        self.lfile = lfile
        self.train = train
        self.split_test = split_test
        self.split_train = split_train
        self.transform_test = transform_test
        self.transform_train = transform_train

        self.loader_input = loader_input
        self.loader_label = loader_label

        if loader_input == 'image':
            self.loader_input = loaders.loader_image
        if loader_input == 'torch':
            self.loader_input = loaders.loader_torch
        if loader_input == 'numpy':
            self.loader_input = loaders.loader_numpy

        if loader_label == 'image':
            self.loader_label = loaders.loader_image
        if loader_label == 'torch':
            self.loader_label = loaders.loader_torch
        if loader_label == 'numpy':
            self.loader_label = loaders.loader_numpy

        if ifile != None:
            imagelist = utils.readtextfile(ifile)
            imagelist = [x.rstrip('\n') for x in imagelist]
        else:
            imagelist = []

        if lfile != None:
            labellist = utils.readtextfile(lfile)
            labellist = [x.rstrip('\n') for x in labellist]
        else:
            labellist = []

        if len(imagelist) == len(labellist):
                shuffle(imagelist, labellist)

        if len(imagelist) > 0 and len(labellist) == 0:
            shuffle(imagelist)

        if len(labellist) > 0 and len(imagelist) == 0:
            shuffle(labellist)

        if (self.split_train < 1.0) & (self.split_train > 0.0):
            if len(imagelist) > 0:
                num = math.floor(self.split*len(imagelist))
                self.images_train = imagelist[0:num]
                self.images_test = images[num+1:len(imagelist)]
            if len(labellist) > 0:
                num = math.floor(self.split*len(labellist))
                self.labels_train = labellist[0:num]
                self.labels_test = labellist[num+1:len(labellist)]

        elif self.split_train == 1.0:
            if len(imagelist) > 0:
                self.images_train = imagelist
            if len(labellist) > 0:
                self.labels_train = labellist

        elif self.split_test == 1.0:
            if len(imagelist) > 0:
                self.images_test = imagelist
            if len(labellist) > 0:
                self.labels_test = labellist

    def __len__(self):
        if self.train == True:
            return len(self.images_train)
        if self.train == False:
            return len(self.images_test)

    def __getitem__(self, index):
        input = {}      
        if self.train == True:
            if len(self.images_train) > 0:
                path = self.images_train[index]
                input['inp'] = self.loader_input(path)

            if len(self.labels_train) > 0:
                path = self.labels_train[index]
                input['tgt'] = self.loader_label(path)

            if self.transform_train is not None:
                input = self.transform_train(input)

            image = input['inp']
            label = input['tgt']

        if self.train == False:
            if len(self.images_test) > 0:
                path = self.images_test[index]
                input['inp'] = self.loader_input(path)

            if len(self.labels_test) > 0:
                path = self.labels_test[index]
                input['tgt'] = self.loader_label(path)

            if self.transform_test is not None:
                input = self.transform_test(input)

            image = input['inp']
            label = input['tgt']

        return image, label
