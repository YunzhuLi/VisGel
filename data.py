from __future__ import print_function, division
import os
import cv2
import torch
import random
from skimage import io, transform
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from utils import resize, crop, adjust_brightness, adjust_saturation
from utils import adjust_contrast, adjust_hue


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)


class VisionTouchDataset(Dataset):

    def __init__(self, phase, data_lst_file,
                 w_timewindow,
                 trans_des=None, trans_lowres=None,
                 trans_to_tensor=None,
                 scale_size=None, crop_size=None,
                 brightness=None, contrast=None, saturation=None, hue=None,
                 loader=default_loader):

        self.phase = phase
        self.recs = open(data_lst_file, 'r').readlines()
        self.w_timewindow = w_timewindow
        self.trans_des = trans_des
        self.trans_lowres = trans_lowres
        self.trans_to_tensor = trans_to_tensor

        self.loader = loader

        self.scale_size = scale_size
        self.crop_size = crop_size
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __len__(self):
        return len(self.recs)

    def variance_of_laplacian(self, image):
        return cv2.Laplacian(image, cv2.CV_64F).var() ** 2

    def cvt_rgb2gray(self, image):
        return cv2.cvtColor(np.array(image).copy(), cv2.COLOR_RGB2GRAY)

    def calc_weight(self, ref_des, des):
        gray_ref = np.array(self.cvt_rgb2gray(ref_des)).astype(np.float)
        gray = np.array(self.cvt_rgb2gray(des)).astype(np.float)
        return self.variance_of_laplacian(gray - gray_ref)

    def get_crop_params(self, phase, img, crop_size):
        w, h = img.size
        th, tw = crop_size, crop_size # hack for now

        if phase == 'train':
            if w == tw and h == th:
                return 0, 0, h, w

            i = random.randint(0, h - th)
            j = random.randint((w - tw) / 2. - 8, (w - tw) / 2. + 8)

        else:
            i = int(round((h - th) / 2.))
            j = int(round((w - tw) / 2.))

        return i, j, th, tw

    def resize_and_crop(self, phase, srcs, scale_size, crop_size):
        len_srcs = len(srcs)

        # resize the images
        for i in range(len_srcs):
            srcs[i] = resize(srcs[i], scale_size)

        crop_params = self.get_crop_params(phase, srcs[0], crop_size)

        # crop the images
        for i in range(len_srcs):
            srcs[i] = crop(srcs[i], crop_params[0], crop_params[1], crop_params[2], crop_params[3])

        return srcs

    def colorjitter(self, srcs, brightness, contrast, saturation, hue):
        len_srcs = len(srcs)

        brightness_factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
        contrast_factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
        saturation_factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
        hue_factor = np.random.uniform(-hue, hue)

        for i in range(len_srcs):
            srcs[i] = adjust_brightness(srcs[i], brightness_factor)
            srcs[i] = adjust_contrast(srcs[i], contrast_factor)
            srcs[i] = adjust_saturation(srcs[i], saturation_factor)
            srcs[i] = adjust_hue(srcs[i], hue_factor)

        return srcs


    def __getitem__(self, idx):
        ref_src, ref_des, src, des, \
        src_pre_0, src_pre_1, \
        src_nxt_0, src_nxt_1 = self.recs[idx].strip().split(" ")

        ref_src = self.loader(ref_src)
        ref_des = self.loader(ref_des)

        src = self.loader(src)
        src_rgb = src.copy()
        des = self.loader(des)

        if self.w_timewindow:
            src_pre_0 = self.loader(src_pre_0)
            src_pre_1 = self.loader(src_pre_1)
            src_nxt_0 = self.loader(src_nxt_0)
            src_nxt_1 = self.loader(src_nxt_1)
            srcs = [ref_src, src, src_pre_0, src_nxt_1, src_pre_1, src_nxt_0]
        else:
            srcs = [ref_src, src]

        # transform src
        srcs = self.resize_and_crop(self.phase, srcs, self.scale_size, self.crop_size)

        if self.phase == 'train':
            srcs = self.colorjitter(srcs, self.brightness, self.contrast,
                                       self.saturation, self.hue)

        srcs_lowres = []
        for i in range(len(srcs)):
            srcs_lowres += [self.trans_lowres(srcs[i])]

        if self.w_timewindow:
            for i in range(1, len(srcs)):
                srcs[i] = self.cvt_rgb2gray(srcs[i])
                srcs_lowres[i] = self.cvt_rgb2gray(srcs_lowres[i])

            ref_src = srcs[0]
            ref_src_lowres = srcs_lowres[0]

            src = np.stack((srcs[1], srcs[2], srcs[3], srcs[4], srcs[5]), axis=-1)
            src_lowres = np.stack((srcs_lowres[1], srcs_lowres[2],
                                      srcs_lowres[3], srcs_lowres[4],
                                      srcs_lowres[5]), axis=-1)
        else:
            ref_src = srcs[0]
            ref_src_lowres = srcs_lowres[0]
            src = srcs[1]
            src_lowres = srcs_lowres[1]

        # transform des
        ref_des = self.trans_des(ref_des)
        ref_des_lowres = self.trans_lowres(ref_des)

        des = self.trans_des(des)
        des_lowres = self.trans_lowres(des)

        # transform all images to torch tensor
        ref_src = self.trans_to_tensor(ref_src)
        ref_src_lowres = self.trans_to_tensor(ref_src_lowres)
        src = self.trans_to_tensor(src)
        src_lowres = self.trans_to_tensor(src_lowres)
        src_rgb = self.trans_to_tensor(src_rgb)

        ref_des = self.trans_to_tensor(ref_des)
        ref_des_lowres = self.trans_to_tensor(ref_des_lowres)
        des = self.trans_to_tensor(des)
        des_lowres = self.trans_to_tensor(des_lowres)

        return ref_src_lowres, ref_des_lowres, src_lowres, des_lowres, \
               ref_src, ref_des, src, src_rgb
