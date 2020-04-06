#!/usr/bin/env python
import os
import numpy as np
from scipy import ndimage
from data_process_func import *
import matplotlib.pyplot as plt


data_root = '/lyc/MICCAI-19-StructSeg/HaN_OAR_center_crop'
filename_list = ['data.nii.gz', 'label.nii.gz']
savename_list = ['crop_data.nii.gz', 'crop_label.nii.gz']
modelist = [ 'valid']
scale_num = [16, 16, 16]
save_as_nifty = True
respacing = False
r = 128
thresh_lis = [-500, -100, 400, 1500]
norm_lis = [0, 0.1, 0.8, 1]
normalize = img_multi_thresh_normalized


for mode in modelist:
    filelist =os.listdir(os.path.join(data_root, mode))
    filenum = len(filelist)
    for ii in range(filenum):
        data_path = os.path.join(data_root, mode, filelist[ii], filename_list[0])
        data_crop_norm_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[0])

        label_path = os.path.join(data_root, mode, filelist[ii], filename_list[1])
        label_crop_save_path = os.path.join(data_root, mode, filelist[ii], savename_list[1])
        if respacing:
            data = load_and_respacing_nifty_volume_as_array(data_path, target_spacing=[3,1,1], order=2)
            label = np.int8(load_and_respacing_nifty_volume_as_array(label_path, mode='label', target_spacing=[3,1,1]))
        else:
            data = load_nifty_volume_as_array(data_path)
            label = np.int8(load_nifty_volume_as_array(label_path))

        center = data.shape[1] // 2

	data_crop = data[:, center-r:center+r, center-r:center+r]
	data_crop_norm = normalize(data_crop, thresh_lis, norm_lis)
	label_crop = label[:, center-r:center+r, center-r:center+r]



        if save_as_nifty:
            save_array_as_nifty_volume(data_crop_norm, data_crop_norm_save_path)
            save_array_as_nifty_volume(label_crop, label_crop_save_path)
        else:
            np.save(data_crop_norm_save_path, data_crop_norm)
            np.save(label_crop_save_path, label_crop)
        print('成功储存', filelist[ii])
