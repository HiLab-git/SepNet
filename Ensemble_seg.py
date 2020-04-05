#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import time
import os
import shutil
import torch.tensor
from util.train_test_func import *
from util.parse_config import parse_config
from util.binary import assd, dc, hd95
from data_process.data_process_func import save_array_as_nifty_volume
from util.assd_evaluation import one_hot
from skimage import morphology

def ensemble():
    config ={
            'data': {
                'data_root': '/lyc/Head-Neck/MICCAI-19-StructSeg/HaN_OAR_center_crop/',
                'save_root': '/lyc/Head-Neck/MICCAI-19-StructSeg/HaN_OAR_center_crop/',
                'seg_name': ['subprob_0.nii.gz','subprob_1.nii.gz','subprob_2.nii.gz','subprob_3.nii.gz','subprob_4.nii.gz','subprob_5.nii.gz'],
                             # , 'subseg_6.nii.gz', 'subseg_7.nii.gz', 'subseg_8.nii.gz'],
                'label_name': 'crop_label.nii.gz',
                'save_name': 'weighted_enseg.nii.gz',
                'class_num': 23
            },
            }
    config_data = config['data']
    Mode = ['valid']
    class_num = config_data['class_num']
    save = True
    delete = False
    cal_dice = False
    cal_hd95 = False

    for mode in Mode:
        patient_list = os.listdir(config_data['data_root']+mode)
        patient_num = len(patient_list)
        dice_array = np.zeros([patient_num, class_num])
        hd95_array = np.zeros([patient_num, class_num])
        for patient_order in range(patient_num):
            patient_path = os.path.join(config_data['data_root'], mode, patient_list[patient_order])
            label_path = os.path.join(patient_path, config_data['label_name'])
            save_path = os.path.join(config_data['save_root'], mode, patient_list[patient_order], config_data['save_name'])
            label = torch.from_numpy(load_nifty_volume_as_array(label_path, transpose=True))
            seg = 0
            for seg_order in range(len(config_data['seg_name'])):
                seg_name = config_data['seg_name'][seg_order]
                seg_path = os.path.join(patient_path, seg_name)
                cur_seg = load_nifty_volume_as_array(seg_path, transpose=False).astype(np.uint16)
                # for ii in range(class_num):
                #     cur_seg[ii] *= weight_0[ii, -seg_order-1]
                seg += cur_seg
                if delete:
                    shutil.rmtree(seg_path)
            seg = np.argmax(seg, axis=0).astype(np.int16)
            onehot_seg = one_hot(seg, class_num)
            onehot_label = one_hot(label, class_num)
            if cal_dice:
                Dice = np.zeros(class_num)
                for i in range(class_num):
                    Dice[i] = dc(onehot_seg[i], onehot_label[i])
                dice_array[patient_order] = Dice
                print('patient order', patient_order, ' dice:', Dice)
            if cal_hd95:
                HD = np.zeros(class_num)
                for i in range(class_num):
                    HD[i] = hd95(onehot_seg[i], onehot_label[i])
                hd95_array[patient_order] = HD
                print('patient order', patient_order, ' dice:', HD)

            if save:
                save_array_as_nifty_volume(seg, save_path, transpose=True)

        if cal_dice:
            dice_array[:, 0] = np.mean(dice_array[:, 1::], 1)
            dice_mean = np.mean(dice_array, 0)
            dice_std = np.std(dice_array, 0)
            print('{0:} mode: mean dice:{1:}, std of dice:{2:}'.format(mode, dice_mean, dice_std))

        if cal_hd95:
            hd95_array[:, 0] = np.mean(hd95_array[:, 1::], 1)
            hd95_mean = np.mean(hd95_array, 0)
            hd95_std = np.std(hd95_array, 0)
            print('{0:} mode: mean dice:{1:}, std of dice:{2:}'.format(mode, hd95_mean, hd95_std))

if __name__ == '__main__':
    ensemble()

