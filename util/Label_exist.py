import os
import numpy as np
import scipy.ndimage
import nibabel
from util.pre_process import *

### 由于肿瘤医院数据标注不全，我们需要另存一个文档指出当前数据具有哪些标注

data_root = '/lyc/MICCAI-19-StructSeg/HaN_OAR/train'
filename = 'crop_label.nii.gz'
savefilename = 'label_exist.npy'
file_list = os.listdir(data_root)
classnum = 23

for file in file_list:
    data_path = os.path.join(data_root, file, filename)
    save_path = os.path.join(data_root, file, savefilename)
    data = load_nifty_volume_as_array(data_path)
    label_exist = np.zeros(classnum)
    for i in range(classnum):
        if np.sum(np.where(data == i)) > 5:
            label_exist[i] = 1
    np.savetxt(save_path, label_exist)
    print('已储存', file)
