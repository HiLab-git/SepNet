#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import time
import os
import shutil
import torch.tensor
from util.train_test_func import *
from util.parse_config import parse_config
from util.binary import assd, dc
from data_process.data_process_func import save_array_as_nifty_volume
from util.assd_evaluation import one_hot
from skimage import morphology

## 计算多模型uncertainty
def uncertain():
    config ={
            'data': {
                'data_root': '/lyc/Head-Neck/MICCAI-19-StructSeg/HaN_OAR_center_crop/',
                'save_root': '/lyc/Head-Neck/MICCAI-19-StructSeg/HaN_OAR_center_crop/',
                'seg_name': ['subseg_0.nii.gz','subseg_1.nii.gz','subseg_2.nii.gz','subseg_3.nii.gz','subseg_4.nii.gz','subseg_5.nii.gz'],
                'label_name': 'crop_label.nii.gz',
                'save_name': 'uncertain.nii.gz',
                'class_num': 23
            },
            }
    config_data = config['data']
    Mode = ['valid']
    class_num = config_data['class_num']
    save = True
    delete = False

    for mode in Mode:
        patient_list = os.listdir(config_data['data_root']+mode)
        patient_num = len(patient_list)
        for patient_order in range(patient_num):
            patient_path = os.path.join(config_data['data_root'], mode, patient_list[patient_order])
            save_path = os.path.join(config_data['save_root'], mode, patient_list[patient_order], config_data['save_name'])
            seg_freq = 0
            for seg_order in range(len(config_data['seg_name'])):
                seg_name = config_data['seg_name'][seg_order]
                seg_path = os.path.join(patient_path, seg_name)
                cur_seg = load_nifty_volume_as_array(seg_path, transpose=True)
                cur_seg = one_hot(cur_seg, class_num).astype(np.float)
                seg_freq += cur_seg
                if delete:
                    shutil.rmtree(seg_path)
            seg_freq /= 6
            uncertain = np.sum(-seg_freq*np.log(seg_freq+0.00001), axis=0)

            print('计算完{0:}'.format(patient_list[patient_order]))
            if save:
                save_array_as_nifty_volume(uncertain, save_path, transpose=True)




if __name__ == '__main__':
	uncertain()
