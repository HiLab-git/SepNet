# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from data_process.data_process_func import *
from data_process.transform import *
from scipy import ndimage
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import os.path
import random
import math
import nibabel
import time
from scipy import ndimage


class LYC_dataset(Dataset):

    def __init__(self, config, stage):
        """
        用于分割的数据集
        """
        self.config    = config
        self.stage = stage
        self.data_root = config['data_root']
        self.batchsize = self.config['batch_size']
        self.subdatashape = self.config['subdata_shape']
        self.testdatashape = self.config['test_data_shape']
        self.classnum = int(self.config['class_num'])
        self.img_name = self.config['img_name']
        self.label_name = self.config['label_name']
        self.label_exist_name = self.config['label_exist_name']
        self.patient_name = os.listdir(self.data_root+'/'+self.stage)
        self.patient_num = len(self.patient_name)
        self.output_feature = self.config['output_feature']
        self.random_rotate = self.config['random_rotate']
        self.random_scale = self.config['random_scale']
        if self.output_feature:
            self.feature_name = self.config['feature_name']
        self.overlap_num = self.config['overlap_num']

    def get_subimage_batch(self):
        '''
        just for train!!!
        :return: images and labels with subshape
        '''
        patientchosen = random.sample(self.patient_name, self.batchsize)
        batch = {}
        databatch = []
        labelbatch = []
        labelexistbatch = []
        multichannel = isinstance(self.img_name, list)
        if multichannel:
            chan_num = len(self.img_name)
            for i in range(self.batchsize):
                datapath = []
                originaldata = []
                labelpath = "{0:}/{1:}/{2:}/{3:}".format(self.data_root, self.stage, patientchosen[i], self.label_name)
                for sub_img_name in self.img_name:
                    subdatapath = "{0:}/{1:}/{2:}/{3:}".format(self.data_root, self.stage, patientchosen[i], sub_img_name)
                    datapath.append(subdatapath)
                if datapath[0].endswith('npy') or datapath[0].endswith('npz'):
                    for subdatapath in datapath:
                        originaldata.append(np.load(subdatapath))
                    originallabel = np.load(labelpath)
                elif datapath[0].endswith('nii') or datapath[0].endswith('nii.gz'):
                    for subdatapath in datapath:
                        originaldata=load_nifty_volume_as_array(subdatapath, transpose=False)
                    originallabel = load_nifty_volume_as_array(labelpath)
                else:
                    ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')

                if self.random_scale:
                    zoomfactor = random.choice([0.8, 1, 1.2])
                    originallabel = ndimage.interpolation.zoom(originallabel, zoomfactor, order = 0)
                    for ii in range(chan_num):
                        originaldata[ii] = ndimage.interpolation.zoom(originaldata[ii], zoomfactor, order = 3)

                originaldata = np.asarray(originaldata)
                shapemin = [random.randint(0, originaldata.shape[ii+1]-self.subdatashape[ii]) for ii in range(3)]
                subdata = originaldata[:, shapemin[0]:shapemin[0]+self.subdatashape[0], shapemin[1]:shapemin[1]+self.subdatashape[1],
                                    shapemin[2]:shapemin[2]+self.subdatashape[2]]
                sublabel = originallabel[shapemin[0]:shapemin[0]+self.subdatashape[0], shapemin[1]:shapemin[1]+self.subdatashape[1],
                                shapemin[2]:shapemin[2]+self.subdatashape[2]]

                if self.random_rotate:
                    subdata,sublabel = random_rotate(subdata, sublabel, p=0.9, degrees=[15, -15], axes=[0, 1, 2])


                databatch.append(subdata)
                labelbatch.append(sublabel)

        else:
            for i in range(self.batchsize):
                datapath = "{0:}/{1:}/{2:}/{3:}".format(self.data_root, self.stage, patientchosen[i], self.img_name)
                labelpath = "{0:}/{1:}/{2:}/{3:}".format(self.data_root, self.stage, patientchosen[i], self.label_name)

                if datapath.endswith('npy') or datapath.endswith('npz'):
                    originaldata = np.load(datapath)
                    originallabel = np.load(labelpath)
                elif datapath.endswith('nii') or datapath.endswith('nii.gz'):
                    originaldata = load_nifty_volume_as_array(datapath)
                    originallabel = load_nifty_volume_as_array(labelpath)
                else:
                    ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')


                if self.random_scale:
                    zoomfactor = random.choice([0.8,1,1.2])
                    originallabel = ndimage.interpolation.zoom(originallabel, zoomfactor, order = 0)
                    originaldata = ndimage.interpolation.zoom(originaldata, zoomfactor, order = 3)


                shapemin = [random.randint(0, originaldata.shape[ii]-self.subdatashape[ii]) for ii in range(3)]
                subdata = originaldata[shapemin[0]:shapemin[0]+self.subdatashape[0], shapemin[1]:shapemin[1]+self.subdatashape[1],
                                shapemin[2]:shapemin[2]+self.subdatashape[2]]
                sublabel = originallabel[shapemin[0]:shapemin[0]+self.subdatashape[0], shapemin[1]:shapemin[1]+self.subdatashape[1],
                                shapemin[2]:shapemin[2]+self.subdatashape[2]]
                if self.random_rotate:
                    subdata,sublabel = random_rotate(subdata, sublabel, p=0.5, degrees=[15, -15], axes=[0, 1, 2])


                databatch.append(subdata[np.newaxis, :])
                labelbatch.append(sublabel)


        batch['images'] = databatch
        batch['labels'] = labelbatch
        return batch



    def get_list_img(self, patient_order):
        '''
        test与valid用,将给定patient的img与label按z轴切片,底端部分会有些重叠
        :patient_order:在patient_name里所选病人的order
        :return: images:N*1*H/N*W*L,N为所切割片数,1是为了满足预测时维数需要所加,相当于预测时batchsize为1
                 labels:1*H*W*L
        '''
        print(self.patient_name[patient_order])
        patient_path = "{0:}/{1:}/{2:}".format(self.data_root, self.stage, self.patient_name[patient_order])
        databatch = []
        batch = {}
        multichannel = isinstance(self.img_name, list)

        if multichannel:
            datapath = []
            originaldata = []
            labelpath = os.path.join(patient_path, self.label_name)
            for sub_img_name in self.img_name:
                subdatapath =os.path.join(patient_path, sub_img_name)
                datapath.append(subdatapath)
            if datapath[0].endswith('npy') or datapath[0].endswith('npz'):
                for subdatapath in datapath:
                    originaldata = np.load(subdatapath)
                originallabel = np.load(labelpath)[np.newaxis, :]
            elif datapath[0].endswith('nii') or datapath[0].endswith('nii.gz'):
                for subdatapath in datapath:
                    originaldata = load_nifty_volume_as_array(subdatapath, transpose=False)
                originallabel = load_nifty_volume_as_array(labelpath)[np.newaxis, :]
            else:
                ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')

            originaldata = np.asarray(originaldata)
            if self.output_feature:
                labelpath = os.path.join(patient_path, self.feature_name)
                originalfeature = np.load(labelpath)
                originaldata = np.concatenate((originaldata, originalfeature), 0)
            batch['originalshape'] = originaldata.shape[1::]



            img_number = int(math.ceil(originaldata.shape[1] / self.testdatashape[0]))  # 在z轴上切成的块数,若为小数会向上取整
            for i in range(img_number-1):
                subdata = originaldata[:, i*self.testdatashape[0]:(i+1)*self.testdatashape[0]][np.newaxis, :]
                subdata = torch.from_numpy(subdata).float()
                databatch.append(subdata)
            subdata = originaldata[:, -self.testdatashape[0]::][np.newaxis, :]
            subdata = torch.from_numpy(subdata).float()
            databatch.append(subdata)
        else:
            datapath = os.path.join(patient_path, self.img_name)
            labelpath = os.path.join(patient_path, self.label_name)
            if datapath.endswith('npy') or datapath.endswith('npz'):
                originaldata = np.load(datapath)
                originallabel = np.load(labelpath)[np.newaxis, :]  # 增加一个batchsize维度
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
            for i in range(img_number - 1):
                subdata = originaldata[i * self.testdatashape[0]:(i + 1) * self.testdatashape[0]][np.newaxis, :][
                          np.newaxis, :]
                subdata = torch.from_numpy(subdata).float()
                databatch.append(subdata)
            subdata = originaldata[-self.testdatashape[0]::][np.newaxis, :][np.newaxis, :]
            subdata = torch.from_numpy(subdata).float()
            databatch.append(subdata)

        batch['images'] = databatch
        batch['labels'] = torch.from_numpy(np.int16(originallabel)).float()
        return batch, patient_path

    def get_list_overlap_img(self, patient_order):
        '''
        test与valid用,将给定patient的img与label按z轴切片,底端部分会有些重叠
        :patient_order:在patient_name里所选病人的order
        :return: images:N*1*H/N*W*L,N为所切割片数,1是为了满足预测时维数需要所加,相当于预测时batchsize为1
                 labels:1*H*W*L
        '''
        print(self.patient_name[patient_order])
        patient_path = "{0:}/{1:}/{2:}".format(self.data_root, self.stage, self.patient_name[patient_order])
        databatch = []
        batch = {}
        multichannel = isinstance(self.img_name, list)

        if multichannel:
            datapath = []
            originaldata = []
            labelpath = os.path.join(patient_path, self.label_name)
            for sub_img_name in self.img_name:
                subdatapath =os.path.join(patient_path, sub_img_name)
                datapath.append(subdatapath)
            if datapath[0].endswith('npy') or datapath[0].endswith('npz'):
                for subdatapath in datapath:
                    originaldata = np.load(subdatapath)
                originallabel = np.load(labelpath)[np.newaxis, :]
            elif datapath[0].endswith('nii') or datapath[0].endswith('nii.gz'):
                for subdatapath in datapath:
                    originaldata = load_nifty_volume_as_array(subdatapath, transpose=False)
                originallabel = load_nifty_volume_as_array(labelpath)[np.newaxis, :]
            else:
                ValueError('please input correct file name! i.e.".nii" ".npy" ".nii.gz"')

            originaldata = np.asarray(originaldata)
            if self.output_feature:
                labelpath = os.path.join(patient_path, self.feature_name)
                originalfeature = np.load(labelpath)
                originaldata = np.concatenate((originaldata, originalfeature), 0)
            batch['originalshape'] = originaldata.shape[1::]



            img_number = int(math.ceil(originaldata.shape[1] / self.testdatashape[0]))  # 在z轴上切成的块数,若为小数会向上取整
            for i in range(img_number-1):
                subdata = originaldata[:, i*self.testdatashape[0]:(i+1)*self.testdatashape[0]][np.newaxis, :]
                subdata = torch.from_numpy(subdata).float()
                databatch.append(subdata)
            subdata = originaldata[:, -self.testdatashape[0]::][np.newaxis, :]
            subdata = torch.from_numpy(subdata).float()
            databatch.append(subdata)
        else:
            datapath = os.path.join(patient_path, self.img_name)
            labelpath = os.path.join(patient_path, self.label_name)
            if datapath.endswith('npy') or datapath.endswith('npz'):
                originaldata = np.load(datapath)
                originallabel = np.load(labelpath)[np.newaxis, :]  # 增加一个batchsize维度
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

            img_number = int(math.ceil(originaldata.shape[0] / self.testdatashape[0]))-self.overlap_num+1  # 在z轴上切成的块数,若为小数会向上取整
            for i in range(img_number - 1):
                subdata = originaldata[i * self.testdatashape[0]:(i + self.overlap_num) * self.testdatashape[0]][np.newaxis, :][
                          np.newaxis, :]
                subdata = torch.from_numpy(subdata).float()
                databatch.append(subdata)
            subdata = originaldata[-self.overlap_num*self.testdatashape[0]::][np.newaxis, :][np.newaxis, :]
            subdata = torch.from_numpy(subdata).float()
            databatch.append(subdata)

        batch['images'] = databatch
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
