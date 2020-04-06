# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from data_process.data_process_func import *
from data_process.transform import *
from scipy import ndimage
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import random
import math
from torchvision import transforms as T
import nibabel
import time
from scipy import ndimage


class Struseg_dataset(Dataset):

    def __init__(self, config, stage):
        """
        用于分割的数据集
        """
        self.config    = config
        self.stage = stage
        self.net_mode = config['net_mode']
        self.data_root = config['data_root']
        self.batchsize = self.config['batch_size']
        self.subdatashape = self.config['subdata_shape']
        self.sublabelshape = self.config['sublabel_shape']
        self.testdatashape = self.config['test_data_shape']
        self.testlabelshape = self.config['test_label_shape']
        self.classnum = int(self.config['class_num'])
        self.img_name = self.config['img_name']
        self.label_name = self.config['label_name']
        self.label_exist_name = self.config['label_exist_name']
        self.patient_path = os.path.join(self.data_root, self.stage)
        self.patient_num = len(self.patient_path)
        self.output_feature = self.config['output_feature']
        self.random_rotate = self.config['random_rotate']
        self.random_scale = self.config['random_scale']
        if self.output_feature:
            self.feature_name = self.config['feature_name']

        data_image = glob.glob(self.patient_path+'/*/'+self.img_name)
        self.data_image = data_image

        mask_image = glob.glob(self.patient_path+'/*/'+self.label_name)
        self.mask_image = mask_image

        if self.net_mode == 'train':
            self.transform = T.Compose(T.RandomRotation(15),
                                       T.RandomCrop(self.subdatashape),
                                       T.ToTensor)

    def __getitem__(self, index):
        data_image_path = self.data_image[index]
        mask_image_path = self.mask_image[index]

        if mask_image_path.endswith('npy') or data_image_path.endswith('npz'):
            originaldata = np.load(data_image_path)
            originallabel = np.load(mask_image_path)
        elif mask_image_path.endswith('nii') or mask_image_path.endswith('nii.gz'):
            originaldata = load_nifty_volume_as_array(data_image_path)
            originallabel = load_nifty_volume_as_array(mask_image_path)
        else:
            ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')

        if self.random_scale:
            zoomfactor = random.choice([0.8, 1, 1.2])
            originallabel = ndimage.interpolation.zoom(originallabel, zoomfactor, order=0)
            originaldata = ndimage.interpolation.zoom(originaldata, zoomfactor, order=3)

        if self.transform:
            image_data = self.transform(originaldata)
            mask_data = self.transform(originallabel)

        return image_data, mask_data

    def __len__(self):
        return len(self.data_image)


    def get_list_img(self, patient_order):
        '''
        test与valid用,将给定patient的img与label按z轴切片,底端部分会有些重叠
        :patient_order:在patient_path里所选病人的order
        :return: images:N*1*H/N*W*L,N为所切割片数,1是为了满足预测时维数需要所加,相当于预测时batchsize为1
                 labels:1*H*W*L
        '''
        print(self.patient_path[patient_order])
        patient_path = "{0:}/{1:}/{2:}".format(self.data_root, self.stage, self.patient_path[patient_order])
        datapath = os.path.join(patient_path, self.img_name)
        labelpath = os.path.join(patient_path, self.label_name)
        databatch = []
        labelbatch = []
        batch = {}

        if datapath.endswith('npy') or datapath.endswith('npz'):
            originaldata = np.load(datapath)
            originallabel = np.load(labelpath)[np.newaxis, :] # 增加一个batchsize维度
        elif datapath.endswith('nii') or datapath.endswith('nii.gz'):
            originaldata = load_nifty_volume_as_array(datapath)
            originallabel = load_nifty_volume_as_array(labelpath)[np.newaxis, :]
        else:
            ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')

        if self.output_feature:
            labelpath = os.path.join(patient_path, self.feature_name)
            originalfeature = np.load(labelpath)
            originaldata = np.concatenate((originaldata, originalfeature), 0)
        batch['originalshape'] = originaldata.shape



        img_number = int(math.ceil(originaldata.shape[0] / self.testdatashape[0]))  # 在z轴上切成的块数,若为小数会向上取整
        for i in range(img_number-1):
            subdata = originaldata[i*self.testdatashape[0]:(i+1)*self.testdatashape[0]][np.newaxis, :][np.newaxis, :]
            subdata = torch.from_numpy(subdata).float()
            databatch.append(subdata)
        subdata = originaldata[-self.testdatashape[0]::][np.newaxis, :][np.newaxis, :]
        subdata = torch.from_numpy(subdata).float()
        databatch.append(subdata)


        batch['images'] = databatch
        batch['listlabels'] = labelbatch
        batch['labels'] = torch.from_numpy(np.int16(originallabel)).float()
        return batch, patient_path


def set_noclass_zero(labelexist, prediction, label):
    '''
    当前患者中未标记的器官对应label置0
    :param labelexist: 记录有哪些器官
    :param label:
    :return:
    '''
    realprediction = torch.max(prediction, 1)[1]
    for i in range(labelexist.shape[0]):
        for classnum in range(1, labelexist.shape[1]):
            if labelexist[i, classnum] == 0:
                a = realprediction[i]==classnum     # prediction中为当前class的位置
                b = label[i]!=0&classnum    # label中其它器官的位置
                a = a*(1-b)
                label[i][a] = classnum
    return label
