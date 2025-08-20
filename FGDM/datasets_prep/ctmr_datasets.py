# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import random

import torch
from torch.utils.data import Dataset
import os
# import SimpleITK as sitk
#import nibabel as nib
import numpy as np
import glob
import torch.utils.data as data
import numpy as np
import os
import cv2
#from skimage import transform
from torchvision import transforms
# import matplotlib.pyplot as plt
import pandas as pd



class CBCTDataset(data.Dataset):
    def __init__(self, csv_file, root_dir, size=None, test=None):
        self.df = pd.read_csv(csv_file)            # CSV에서 파일명 리스트 불러옴
        self.filenames = self.df['filename'].tolist()
        self.root_dir = root_dir                   # npy 파일들이 들어 있는 상위 폴더
        self.size = size
        self.test = test

    def __getitem__(self, index):
        filename = self.filenames[index]
        filepath = os.path.join(self.root_dir, filename)

        npimg = np.load(filepath).astype(np.float32)
        # npimg = npimg.astype(np.float32)

        nplabs = (npimg - npimg.min()) / (npimg.max() - npimg.min())
        nplabs = nplabs * 255

        if self.test==True:
            window_size=9
            bilater_random1 = 75
        else:
            window_size = random.randint(1,15)
            bilater_random1 = random.randint(60, 120)
        nplabs = cv2.bilateralFilter(nplabs, window_size, bilater_random1, bilater_random1)

        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        nplabs = nplabs * 255
        nplabs = np.uint8(nplabs)

        x = cv2.Sobel(nplabs, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(nplabs, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        nplabs = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        nplabs = (nplabs - nplabs.min()) / (nplabs.max() - nplabs.min())
        # npimg = npimg.astype(np.float32)
        # nplabs = nplabs.astype(np.float32)

        # nplabs = nplabs.astype(np.float32)
        npimg = torch.from_numpy(np.expand_dims(npimg, 0)).float()
        nplabs = torch.from_numpy(np.expand_dims(nplabs, 0)).float()

        resize = transforms.Resize([self.size, self.size])
        npimg = resize(npimg)
        nplabs = resize(nplabs)

        return npimg, nplabs

    def __len__(self):
        return len(self.filenames)


