import os
import numpy as np
import nibabel
from data_process_func import *

ori_label = [14, 15, 16, 17, 18, 19, 21, 22]
data_root = '/lyc/MICCAI-19-StructSeg/HaN_OAR_normal_spacing'
data_mode = ['train', 'valid']


ori_labelfile_name = 'crop_label.nii.gz'
new_labelfile_name = 'part_crop_label.nii.gz'


for mode in data_mode:
    cur_data_root = os.path.join(data_root, mode)
    file_list = os.listdir(cur_data_root)

    for file in file_list:
        cur_label_path = os.path.join(cur_data_root, file, ori_labelfile_name)
        new_label_path = os.path.join(cur_data_root, file, new_labelfile_name)

        cur_label, spacing = load_nifty_volume_as_array(cur_label_path, return_spacing=True)
        new_label = np.zeros_like(cur_label)
        for i in range(len(ori_label)):
            mask = np.where(cur_label==ori_label[i])
            new_label[mask] = i+1
        save_array_as_nifty_volume(new_label, new_label_path, spacing)
        print('successfully proceed {0:}'.format(file))