# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import scipy
from scipy import ndimage
from PIL import Image
import numpy as np
from graphviz import Digraph
import torch
from torch.autograd import Variable
from visdom import Visdom

def add_countor(In, Seg, Color=(0, 255, 0)):
    Out = In.copy()
    [H, W] = In.size
    for i in range(H):
        for j in range(W):
            if(i==0 or i==H-1 or j==0 or j == W-1):
                if(Seg.getpixel((i,j))!=0):
                    Out.putpixel((i,j), Color)
            elif(Seg.getpixel((i,j))!=0 and  \
                 not(Seg.getpixel((i-1,j))!=0 and \
                     Seg.getpixel((i+1,j))!=0 and \
                     Seg.getpixel((i,j-1))!=0 and \
                     Seg.getpixel((i,j+1))!=0)):
                     Out.putpixel((i,j), Color)
    return Out

def add_segmentation(img, seg, Color=(0, 255, 0)):
    seg = np.asarray(seg)
    if(img.size[1] != seg.shape[0] or img.size[0] != seg.shape[1]):
        print('segmentation has been resized')
    seg = scipy.misc.imresize(seg, (img.size[1], img.size[0]), interp='nearest')
    strt = ndimage.generate_binary_structure(2, 1)
    seg = np.asarray(ndimage.morphology.binary_opening(seg, strt), np.uint8)
    seg = np.asarray(ndimage.morphology.binary_closing(seg, strt), np.uint8)

    img_show = add_countor(img, Image.fromarray(seg), Color)
    strt = ndimage.generate_binary_structure(2, 1)
    seg = np.asarray(ndimage.morphology.binary_dilation(seg, strt), np.uint8)
    img_show = add_countor(img_show, Image.fromarray(seg), Color)
    return img_show


class loss_visualize(object):
    def __init__(self, class_num, env='loss'):
        self.viz = Visdom(env=env)
        epoch = 0
        self.loss = self.viz.line(X=np.array([epoch]),
                                  Y=np.zeros([1, class_num+1]),  # +1是因为除去背景还有train与test
                                  opts=dict(showlegend=True))

    def plot_loss(self, epoch, batch_dice):
        train_dice_mean = np.asarray([batch_dice[1][1::].mean(axis=0)])
        valid_dice_classes = batch_dice[0][1::]
        valid_dice_mean = np.asarray([valid_dice_classes.mean(axis=0)])
        dice = np.append(np.append(train_dice_mean,
                               valid_dice_mean), valid_dice_classes)[np.newaxis, :]
        self.viz.line(
            X=np.array([epoch+1]),
            Y=dice,
            win=self.loss,  # win要保持一致
            update='append')