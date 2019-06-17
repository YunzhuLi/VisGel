from __future__ import print_function
import argparse
import os
import sys
import cv2
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.optim import lr_scheduler
from utils import read_weights
from matplotlib import pyplot as plt

from models import _netG_resnet, _netD, GANLoss
from data import VisionTouchDataset
import imageio


parser = argparse.ArgumentParser()
parser.add_argument('--branch', default='demo', help="demo|eval")
parser.add_argument('--direction', default='vision2touch', help="vision2touch|touch2vision")
parser.add_argument('--epoch', type=int, default=74)
parser.add_argument('--eval_lst', default='data_lst')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--workers', type=int, default=10)
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--np', type=int, default=7)

parser.add_argument('--scale_size', type=int, default=272)
parser.add_argument('--crop_size', type=int, default=256)
parser.add_argument('--scale_size_lowres', type=int, default=128)
parser.add_argument('--crop_size_lowres', type=int, default=128)
parser.add_argument('--brightness', type=float, default=0.)
parser.add_argument('--contrast', type=float, default=0.)
parser.add_argument('--saturation', type=float, default=0.)
parser.add_argument('--hue', type=float, default=0.)

parser.add_argument('--img_dim', type=int, default=128)
parser.add_argument('--split_dim', type=int, default=4)

parser.add_argument('--w_timewindow', type=int, default=1)
parser.add_argument('--w_resampling', type=int, default=1)
parser.add_argument('--w_L1Loss', type=int, default=1)
parser.add_argument('--w_GANLoss', type=int, default=1)

parser.add_argument('--INF', type=int, default=1e8)
parser.add_argument('--dis_thresh', type=int, default=8)

args = parser.parse_args()

# evaluation list
args.eval_lst = os.path.join(
    args.eval_lst, args.branch + '_' + args.direction + '.txt')

# directory
args.dir = os.path.join('dump_' + args.direction)

print(args)

trans_touch = transforms.Compose([
    transforms.Scale(args.crop_size),
    transforms.CenterCrop(args.crop_size),
])

trans_lowres = transforms.Compose([
    transforms.Scale(args.scale_size_lowres),
    transforms.CenterCrop(args.crop_size_lowres)
])

trans_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


img_datasets = {'valid': VisionTouchDataset(
    'valid', args.eval_lst, args.w_timewindow,
    trans_touch, trans_lowres, trans_to_tensor,
    args.scale_size, args.crop_size, args.brightness, args.contrast,
    args.saturation, args.hue)}

dataset_sizes = {x: len(img_datasets[x]) for x in ['valid']}

dataloaders = {x: torch.utils.data.DataLoader(
        img_datasets[x], batch_size=args.batch_size,
        shuffle=True if x == 'train' else False,
        num_workers=args.workers)
        for x in ['valid']}


use_gpu = torch.cuda.is_available()
# use_gpu = False

# define generation network
netG = _netG_resnet(args.np, args.nc, args.ngf, args.w_timewindow)
print(netG)

model_file = os.path.join(args.dir, 'ckps', 'netG_epoch' + str(args.epoch) + '.pth')
print("Load ckp %s" % model_file)
netG.load_state_dict(torch.load(model_file))
netG.eval()

if use_gpu:
    netG.cuda()

# read valid lst
valid_all_lst = open(args.eval_lst, 'r').readlines()
len_valid_all_lst = len(valid_all_lst)



# combine images for simultaneous vision and touch
def combine_three(src, des, des_fake):

    height = 256
    width = 320

    f = height / src.shape[0]
    src = cv2.resize(src, None, fx=f, fy=f)
    offset = (src.shape[1] - width) // 2
    if offset > 0:
        src = src[:, offset:-offset]

    s = args.split_dim
    d = (height - s) // 2

    des = cv2.resize(des, (d, d), interpolation=cv2.INTER_CUBIC)
    des_fake = cv2.resize(des_fake, (d, d), interpolation=cv2.INTER_CUBIC)

    cv2.rectangle(des, (0, 0), (d, d), (0, 255, 0), 6)
    cv2.rectangle(des_fake, (0, 0), (d, d), (255, 0, 0), 6)

    image = np.ones((height, width + d + s, 3), dtype=np.uint8) * 255

    image[:, :width] = src
    image[:d, width+s:] = des
    image[d+s:, width+s:] = des_fake

    return [image]


def denormalize(img):
    return (img * 0.5 + 0.5) * 255


def calc_dis(a, b):
    return np.sqrt(np.sum(np.square(np.array(a) - np.array(b))))


def calc_centers(img):
    img = np.uint8(img)
    img = cv2.GaussianBlur(img, (11, 11), 0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    img = np.uint8(opening)

    contours, _ = cv2.findContours(img.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1)
    centers = []
    for i in range(len(contours)):
        if len(contours[i]) < 3:
            continue
        else:
            moments = cv2.moments(contours[i])
            centers.append((moments['m10']/moments['m00'], moments['m01']/moments['m00']))

        # cv2.circle(img, (int(centers[-1][0]), int(centers[-1][1])), 1, (0, 0, 0), -1)

    # plt.imshow(img)
    # plt.show()

    return centers


def calc_deform(cur, pre, deforms):
    ret_centers = []
    ret_deforms = []
    for i in range(len(pre)):
        dis_min = args.INF
        for j in range(len(cur)):
            dis = calc_dis(pre[i], cur[j])
            if dis < dis_min:
                dis_min = dis
                deform = (pre[i][0] - cur[j][0], pre[i][1] - cur[j][1])
                idx = j

        if dis_min > args.dis_thresh:
            continue

        ret_centers.append(cur[idx])
        ret_deforms.append((deforms[i][0] + deform[0], deforms[i][1] + deform[1]))

    return ret_centers, ret_deforms


def calc_deform_field(centers, centers_gt, deforms, deforms_gt):
    centers_ori = []
    centers_ori_gt = []
    for i in range(len(centers)):
        centers_ori.append((deforms[i][0] + centers[i][0], deforms[i][1] + centers[i][1]))
    for i in range(len(centers_gt)):
        centers_ori_gt.append((deforms_gt[i][0] + centers_gt[i][0], deforms_gt[i][1] + centers_gt[i][1]))

    matched = 0
    ret = 0.
    for i in range(len(centers_ori_gt)):
        dis_min = args.INF
        for j in range(len(centers_ori)):
            dis = calc_dis(centers_ori_gt[i], centers_ori[j])
            if dis < dis_min:
                dis_min = dis
                deform = (deforms_gt[i][0] - deforms[j][0], deforms_gt[i][1] - deforms[j][1])

        if dis_min > args.dis_thresh:
            continue

        ret += np.sqrt(np.square(deform[0]) + np.square(deform[1]))
        matched += 1

    return ret / matched


def calc_deform_seq(fake, des):
    errors = 0.

    centers_lst = []
    deforms_lst = []

    for i in range(args.batch_size):
        centers = calc_centers(cv2.resize(des[i], (200, 200)))

        if i == 0:
            deforms = []
            for j in range(len(centers)):
                deforms.append((0, 0))
        else:
            centers, deforms = calc_deform(centers, pre_centers, deforms)

        centers_lst.append(centers)
        deforms_lst.append(deforms)
        pre_centers = centers

    for i in range(args.batch_size):
        centers = calc_centers(cv2.resize(fake[i], (200, 200)))

        if i == 0:
            deforms = []
            for j in range(len(centers)):
                deforms.append((0, 0))
        else:
            centers, deforms = calc_deform(centers, pre_centers, deforms)

        errors += calc_deform_field(centers, centers_lst[i], deforms, deforms_lst[i])

        pre_centers = centers

    return errors / args.batch_size


if args.branch == 'demo':
    deform_error = 0.
elif args.branch == 'eval':
    deform_error_seen = 0.
    deform_error_unseen = 0.
else:
    raise AssertionError("Unknown branch " + args.branch)

# make folder to store the generated imgs
img_dir = os.path.join(args.dir, args.branch, 'imgs', 'epoch_' + str(args.epoch).zfill(3))
os.system('mkdir -p ' + img_dir)

# make folder to store the generated gifs
action_dir = os.path.join(args.dir, args.branch, 'actions/epoch_' + str(args.epoch).zfill(3))
os.system('mkdir -p ' + action_dir)

for phase in ['valid']:
    for i, data in enumerate(dataloaders[phase], 0):

        volatile = phase == 'valid'

        if use_gpu:
            for j in range(len(data)):
                data[j] = Variable(data[j].cuda(), volatile=volatile)
        else:
            for j in range(len(data)):
                data[j] = Variable(data[j], volatile=volatile)

        ref_src_lowres, ref_des_lowres, src_lowres, des_lowres, \
        ref_src, ref_des, src, src_rgb = data

        fake_des_lowres = netG(ref_src, ref_des, src)

        ref_src_lowres = ref_src_lowres.data.cpu().numpy().transpose((0, 2, 3, 1))
        src_lowres = src_lowres.data[:, :3, :, :].cpu().numpy().transpose((0, 2, 3, 1))
        des_lowres = des_lowres.data.cpu().numpy().transpose((0, 2, 3, 1))
        fake_des_lowres = fake_des_lowres.data.cpu().numpy().transpose((0, 2, 3, 1))

        src_rgb = src_rgb.data.cpu().numpy().transpose((0, 2, 3, 1))

        images = []
        images_only_groundt = []
        images_only_predict = []
        ref_src_lowres = np.uint8(denormalize(ref_src_lowres))
        src_lowres = np.uint8(denormalize(src_lowres))
        des_lowres = np.uint8(denormalize(des_lowres))
        fake_des_lowres = np.uint8(denormalize(fake_des_lowres))

        src_rgb = np.uint8(denormalize(src_rgb))

        png_dir = os.path.join(img_dir, 'rec_' + str(i).zfill(3))
        os.system('mkdir -p ' + png_dir)
        for j in range(args.batch_size):
            imageio.imwrite(os.path.join(png_dir, 'img_' + str(i * args.batch_size + j).zfill(5) + '_rgb.png'), src_rgb[j])
            imageio.imwrite(os.path.join(png_dir, 'img_' + str(i * args.batch_size + j).zfill(5) + '_ref_src.png'), ref_src_lowres[j])
            imageio.imwrite(os.path.join(png_dir, 'img_' + str(i * args.batch_size + j).zfill(5) + '_src.png'), src_lowres[j])
            imageio.imwrite(os.path.join(png_dir, 'img_' + str(i * args.batch_size + j).zfill(5) + '_des.png'), des_lowres[j])
            imageio.imwrite(os.path.join(png_dir, 'img_' + str(i * args.batch_size + j).zfill(5) + '_fake_des.png'), fake_des_lowres[j])
            images += combine_three(src_rgb[j], des_lowres[j], fake_des_lowres[j])

        if args.direction == 'vision2touch':
            imageio.mimsave(os.path.join(action_dir, 'valid_' + str(i).zfill(4) + '.gif'), images, duration=0.05)

            error = calc_deform_seq(des_lowres[..., ::-1], fake_des_lowres[..., ::-1])
            print("processed", i * args.batch_size, '/', len(dataloaders[phase]) * args.batch_size, 'deform:', error)

            if args.branch == 'demo':
                deform_error += error
            elif args.branch == 'eval':
                if i < len(dataloaders[phase]) // 2:
                    deform_error_seen += error
                else:
                    deform_error_unseen += error

        else:
            imageio.mimsave(os.path.join(action_dir, 'valid_' + str(i).zfill(4) + '.gif'), images, duration=1)
            print("processed", i * args.batch_size, '/', len(dataloaders[phase]) * args.batch_size)


if args.direction == 'vision2touch':
    if args.branch == 'demo':
        print("Deform Error:", deform_error / len(dataloaders['valid']))
    elif args.branch == 'eval':
        print("Deform Error Seen:", deform_error_seen / (len(dataloaders['valid']) // 2))
        print("Deform Error Unseen:", deform_error_unseen / (len(dataloaders['valid']) // 2))
        print("Total Deform Error:", (deform_error_seen + deform_error_unseen) / len(dataloaders['valid']))

