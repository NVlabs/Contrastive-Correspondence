# ------------------------------------------------------------------------------
# MIT License
# Copyright (c) 2019 UVC
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# Code is derived from https://github.com/Liusifei/UVC/blob/master/libs/loader.py
# Modified by Taihong Xiao
# ------------------------------------------------------------------------------

import numpy as np
import cv2
import time
import random
from PIL import Image
import torch
from os.path import exists, join, split
import glob
import libs.transforms_multi as transforms
from torchvision import datasets
import os
import sys


def framelist_loader(video_path, sample_num):
    cap = cv2.VideoCapture(video_path)
    video_len = int(cap.get(7))
    if video_len < sample_num:
        print("The number of video frames of {} is less than {}, skip to the next".format(
            video_path, sample_num))
        return []

    frame_list = []

    # ensure that sampled frames are within 50 frames
    if video_len > 50:
        start = np.random.randint(0, video_len-50)
        end = start + 50
    else:
        start = 0
        end = video_len
    id_list = sorted(np.random.choice(
        range(start, end), sample_num, replace=False))

    for ii in range(sample_num):
        cap.set(1, id_list[ii])
        success, image = cap.read()
        if not success:
            print('Error while reading video {}, id {}, end {}, video_len {}:'.format(
                video_path, id_list[ii], end, video_len))
            return []

        h, w, _ = image.shape
        h = (h // 64) * 64
        w = (w // 64) * 64

        image = cv2.resize(image, (w, h))
        image = image.astype(np.uint8)
        pil_im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_list.append(pil_im)
    return frame_list


class VidListCycleFast(torch.utils.data.Dataset):
    # for cycle training
    def __init__(self, video_path, list_path, full_size, patch_size, frame_num, rotate=10):
        super(VidListCycleFast, self).__init__()
        self.data_dir = video_path
        self.list_path = list_path
        self.full_size = full_size
        self.patch_size = patch_size
        self.frame_num = frame_num
        self.rotate = rotate

        normalize = transforms.Normalize(
            mean=(128, 128, 128), std=(128, 128, 128))

        self.transforms = transforms.Compose([
            transforms.ResizeandPad(full_size),
            transforms.ToTensor(),
            normalize
        ])
        self.read_list()

    def __getitem__(self, idx):
        frame_list = []
        while len(frame_list) == 0:
            video = self.list[idx]
            frame_list = framelist_loader(video, self.frame_num)
            idx = (idx + 1) % len(self)

        frames = torch.stack(self.transforms(*frame_list), 0)
        return frames

    def __len__(self):
        return len(self.list)

    def read_list(self):
        path = join(self.list_path)
        root = path.partition("Kinetices/")[0]
        if not exists(path):
            raise Exception(
                "{} does not exist in kinet_dataset.py.".format(path))
        self.list = [line.replace("/Data/", root).strip()
                     for line in open(path, 'r')]


class VidListCycle(torch.utils.data.Dataset):
    # for cycle training
    def __init__(self, video_path, list_path, full_size, patch_size, frame_num, rotate=10):
        super(VidListCycle, self).__init__()
        self.data_dir = video_path
        self.list_path = list_path
        self.full_size = full_size
        self.patch_size = patch_size
        self.frame_num = frame_num
        self.rotate = rotate

        normalize = transforms.Normalize(
            mean=(128, 128, 128), std=(128, 128, 128))

        self.transforms1 = transforms.Compose([
            transforms.ResizeandPad(full_size),
        ])

        self.transforms2 = transforms.Compose([
            transforms.RandomRotate(rotate),
            transforms.RandomCrop(patch_size),
        ])
        self.transforms3 = transforms.Compose([
            transforms.ToTensor(),
            normalize])

        self.read_list()

    def __getitem__(self, idx):
        frame_list = []
        while len(frame_list) == 0:
            video = self.list[idx]
            frame_list = framelist_loader(video, self.frame_num)
            idx += 1

        frame_list = self.transforms1(*frame_list)

        ww, hh = np.meshgrid(
            np.arange(frame_list[0].shape[1]), np.arange(frame_list[0].shape[0]))
        grid = np.stack([ww / float(frame_list[0].shape[1]),
                        hh / float(frame_list[0].shape[0])], -1)

        crop, crop_gt = self.transforms2(frame_list[0], grid)

        crop_gt = cv2.resize(crop_gt.astype(
            np.float32), (crop_gt.shape[1] // 8, crop_gt.shape[0] // 8), cv2.INTER_NEAREST)
        crop_gt = np.stack([crop_gt[..., 0] * frame_list[0].shape[1],
                           crop_gt[..., 1] * frame_list[0].shape[0]], -1)
        crop_gt = torch.from_numpy(crop_gt.copy()).contiguous().float()

        crop = self.transforms3(crop)[0]
        frames = torch.stack(self.transforms3(*frame_list), 0)

        return crop, crop_gt, frames

    def __len__(self):
        return len(self.list)

    def read_list(self):
        path = join(self.list_path)
        root = path.partition("Kinetices/")[0]
        if not exists(path):
            raise Exception(
                "{} does not exist in kinet_dataset.py.".format(path))
        self.list = [line.replace("/Data/", root).strip()
                     for line in open(path, 'r')]


class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(ZipDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.indices = []
        for i, d in enumerate(self.datasets):
            # assert not isinstance(d, IterableDataset), "ZipDataset does not support IterableDataset"
            self.indices.append(torch.randperm(len(self)) % len(d))

    def __len__(self):
        return max(list(map(len, self.datasets)))

    def __getitem__(self, idx):
        return tuple(d[self.indices[i][idx]] for i, d in enumerate(self.datasets))


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, imagenet_dataset, pf_dataset):
        super(ConcatDataset, self).__init__()
        self.imagenet_dataset = imagenet_dataset
        self.pf_dataset = pf_dataset

    def __len__(self):
        return len(self.imagenet_dataset) + len(self.pf_dataset)

    def __getitem__(self, idx):
        if idx < len(self.imagenet_dataset):
            # return the (im_q, im_k) pair
            return self.imagenet_dataset[idx][0]
        else:
            data = self.pf_dataset[idx-len(self.imagenet_dataset)]
            return (data['moco_img1'], data['moco_img2'])
