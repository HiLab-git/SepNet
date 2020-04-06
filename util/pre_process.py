# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os    
import nibabel
import numpy as np
import random
#import vtk
#from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

def get_roi_size(inputVolume):
    [d_idxes, h_idxes, w_idxes] = np.nonzero(inputVolume)
    mind = d_idxes.min(); maxd = d_idxes.max()
    minh = h_idxes.min(); maxh = h_idxes.max()
    minw = w_idxes.min(); maxw = w_idxes.max()
    return [maxd - mind, maxh - minh, maxw - minw]

def get_unique_image_name(img_name_list, subname):
    img_name = [x for x in img_name_list if subname in x]
    assert(len(img_name) == 1)
    return img_name[0]

def load_nifty_volume_as_array(filename):
    # input shape [W, H, D]
    # output shape [D, H, W]
    img = nibabel.load(filename)
    data = img.get_data()
    data = np.transpose(data, [2,1,0])
    return data

#def load_vtk_volume_as_array(imgName):
#    if(imgName.endswith('nii')):
#        reader=vtk.vtkNIFTIImageReader()
#    elif(imgName.endswith('mha')):
#        reader = vtk.vtkMetaImageReader()
#    else:
#        raise ValueError('could not open file {0:}'.format(imgName))
#    reader.SetFileName(imgName)
#    reader.Update()
#    vtkImg =  reader.GetOutput()   
#    shape = vtkImg.GetDimensions()
#    sc = vtkImg.GetPointData().GetScalars()
#    img_np = np.array(vtk_to_numpy(sc).reshape([shape[2],shape[1],shape[0]]))
#    return img_np

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2, 1, 0])
    img = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(img, filename)

# for brats 17
def load_all_modalities_in_one_folder(patient_dir, ground_truth = True):
    img_name_list = os.listdir(patient_dir)
    img_list = []
    sub_name_list = ['flair.nii', 't1ce.nii', 't1.nii', 't2.nii']
    if(ground_truth):
        sub_name_list.append('seg.nii')
    for sub_name in sub_name_list:
        img_name = get_unique_image_name(img_name_list, sub_name)
        img   = load_nifty_volume_as_array(os.path.join(patient_dir, img_name))
        img_list.append(img)
    return img_list

# for brats15
def load_all_modalities_in_one_folder_15(patient_dir, ground_truth = True):
    img_name_list = os.listdir(patient_dir)
    print('image names', img_name_list)
    img_list = []
    sub_name_list = ['Flair.', 'T1.', 'T1c.', 'T2.']
    if(ground_truth):
        sub_name_list.append('OT.')
    for sub_name in sub_name_list:
        for img_name in img_name_list:
            if(sub_name in img_name):
                full_img_name = patient_dir + '/' + img_name + '/' + img_name + '.mha'
                print(full_img_name)
                img   = load_vtk_volume_as_array(full_img_name)
                img_list.append(img)
    return img_list

def get_itensity_statistics(volume, n_pxl, iten_sum, iten_sq_sum):
    volume = np.asanyarray(volume, np.float32)
    pixels = volume[volume > 0]
    n_pxl = n_pxl + len(pixels)
    iten_sum = iten_sum + pixels.sum()
    iten_sq_sum = iten_sq_sum + np.square(pixels).sum()
    return n_pxl, iten_sum, iten_sq_sum

def get_all_patients_dir(data_root):
    sub_sets = ['HGG/', 'LGG/']
    all_patients_list = []
    for sub_source in sub_sets:
        sub_source = data_root + sub_source
        patient_list = os.listdir(sub_source)
        patient_list = [sub_source + x for x in patient_list if 'Brats' in x]
        all_patients_list.extend(patient_list)
        print('patients for ', sub_source,len(patient_list))
    print("total patients ", len(all_patients_list))
    return all_patients_list

def get_roi_range_in_one_dimention(x0, x1, L):
    margin = L - (x1 - x0)
    mg0 = margin/2
    mg1 = margin - mg0
    x0 = x0 - mg0
    x1 = x1 + mg1
    return [x0, x1]

def get_roi_from_volumes(volumes):
    [outD, outH, outW] = [144, 176, 144]
    [d_idxes, h_idxes, w_idxes] = np.nonzero(volumes[0])
    mind = d_idxes.min(); maxd = d_idxes.max()
    minh = h_idxes.min(); maxh = h_idxes.max()
    minw = w_idxes.min(); maxw = w_idxes.max()
    print(mind, maxd, minh, maxh, minw, maxw)
    [mind, maxd] = get_roi_range_in_one_dimention(mind, maxd, outD)
    [minh, maxh] = get_roi_range_in_one_dimention(minh, maxh, outH)
    [minw, maxw] = get_roi_range_in_one_dimention(minw, maxw, outW)
    print(mind, maxd, minh, maxh, minw, maxw)
    roi_volumes = []
    for volume in volumes:
        roi_volume = volume[np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
        roi_volumes.append(roi_volume)
        print(roi_volume.shape)
    return roi_volumes, [mind, maxd, minh, maxh, minw, maxw]

def get_training_set_statistics(): 
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/BRATS17TrainingData/'
    all_patients_list = get_all_patients_dir(source_root)

    # get itensity mean and std
#     n_pxls = np.zeros([4], np.float32)
#     iten_sum = np.zeros([4], np.float32)
#     iten_sq_sum = np.zeros([4], np.float32)
#     for patient_dir in all_patients_list:
#         volumes = load_all_modalities_in_one_folder(patient_dir)
#         for i in range(4):
#             n_pxls[i], iten_sum[i], iten_sq_sum[i] = get_itensity_statistics(
#                     volumes[i], n_pxls[i], iten_sum[i], iten_sq_sum[i])
#         print patient_dir
#         print volumes[0][volumes[0]>0].mean(), volumes[1][volumes[1]>0].mean(), volumes[2][volumes[2]>0].mean(), volumes[3][volumes[3]>0].mean()
#     mean = np.divide(iten_sum, n_pxls)
#     sq_men = np.divide(iten_sq_sum, n_pxls)
#     std = np.sqrt(sq_men - np.square(mean))
#     print mean, std    
   
    roi_size = []
    for patient_dir in all_patients_list:
        volumes = load_all_modalities_in_one_folder(patient_dir)
        for i in range(4):
            roi = get_roi_size(volumes[i])
            roi_size.append(roi)
    roi_size = np.asarray(roi_size)
    print(roi_size.mean(axis = 0), roi_size.std(axis = 0))
        
def extract_roi_for_training_set():
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/BRATS17TrainingData/'
    target_root = 'Training_extract'
    sub_sets = ['HGG/', 'LGG/']
    modality_names = ['flair.nii.gz', 't1ce.nii.gz', 't1.nii.gz', 't2.nii.gz', 'seg.nii.gz']
    all_patients_list = get_all_patients_dir(source_root)
    for patient_dir in all_patients_list:
        volumes = load_all_modalities_in_one_folder(patient_dir)
        roi_volumes, roi = get_roi_from_volumes(volumes)
        for i in range(len(roi_volumes)):
            save_patient_dir = patient_dir.replace("BRATS17TrainingData", target_root)
            print(save_patient_dir)
            if(not os.path.isdir(save_patient_dir)):
                os.mkdir(save_patient_dir)
            save_name = os.path.join(save_patient_dir, modality_names[i])
            img = nibabel.Nifti1Image(roi_volumes[i], np.eye(4))
            nibabel.save(img, save_name)
            
def split_data(split_name, seed):
    source_root = '/Users/guotaiwang/Documents/data/BRATS2017/Training_extract/'
    all_patients_list =  get_all_patients_dir(source_root) 
    random.seed(seed)
    n = len(all_patients_list)
    n_test = 50
    test_mask = np.zeros([n])
    test_idx = random.sample(range(n), n_test)
    test_mask[test_idx] = 1
    
    train_list = []  
    test_list  = [] 
    for i in range(n): 
        patient_split = all_patients_list[i].split('/')
        patient = patient_split[-2] + '/' + patient_split[-1]
        if(test_mask[i]):
            test_list.append(patient) 
        else:
            train_list.append(patient)
    print("train_list", len(train_list))
    print("test_list ", len(test_list))
    train_file = open(split_name + '/train.txt', 'w')
    for patient in train_list:
        train_file.write("%s\n" % patient)
    test_file = open(split_name + '/test.txt', 'w')
    for patient in test_list:
        test_file.write("%s\n" % patient)  
    seed_file =  open(split_name + '/seed.txt', 'w')
    seed_file.write("%d"%seed)
    
def Brats17_data_set_crop_rename(source_folder, save_folder, crop, ground_truth):
    patient_list = os.listdir(source_folder)
    patient_list = [x for x in patient_list if 'Brats17' in x]
    margin = 5
    save_postfix = ['Flair', 'T1c', 'T1', 'T2']
    if(ground_truth):
        save_postfix.append('Label')
    print('patient number ', len(patient_list))
    for patient_dir in patient_list:
        print(patient_dir)
        continue
        full_patient_dir = os.path.join(source_folder, patient_dir)
        imgs = load_all_modalities_in_one_folder(full_patient_dir, ground_truth = ground_truth)
        assert(len(imgs)  == len(save_postfix))
        if(crop):
            [d_idxes, h_idxes, w_idxes] = np.nonzero(imgs[0])
            mind = d_idxes.min() - margin; maxd = d_idxes.max() + margin
            minh = h_idxes.min() - margin; maxh = h_idxes.max() + margin
            minw = w_idxes.min() - margin; maxw = w_idxes.max() + margin
        for mod_idx in range(len(save_postfix)):
            if(crop):
                roi_volume = imgs[mod_idx][np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
            else:
                roi_volume = imgs[mod_idx]
            save_name = "{0:}_{1:}.nii.gz".format(patient_dir, save_postfix[mod_idx])
            save_name = os.path.join(save_folder, save_name)
            save_array_as_nifty_volume(roi_volume, save_name)

def Brats15_data_set_crop_rename(source_folder, save_folder, crop):
    patient_list = os.listdir(source_folder)
    patient_list = [x for x in patient_list if 'brats' in x]
    margin = 5
    save_postfix = ['Flair', 'T1', 'T1c', 'T2', 'Label']
    for patient_dir in patient_list:
        print(patient_dir)
        full_patient_dir = os.path.join(source_folder, patient_dir)
        imgs = load_all_modalities_in_one_folder_15(full_patient_dir, ground_truth = True)
        assert(len(imgs)  == len(save_postfix))
        if(crop):
            [d_idxes, h_idxes, w_idxes] = np.nonzero(imgs[0])
            mind = d_idxes.min() - margin; maxd = d_idxes.max() + margin
            minh = h_idxes.min() - margin; maxh = h_idxes.max() + margin
            minw = w_idxes.min() - margin; maxw = w_idxes.max() + margin
        for mod_idx in range(len(imgs)):
            if(crop):
                roi_volume = imgs[mod_idx][np.ix_(range(mind, maxd), range(minh, maxh), range(minw, maxw))]
            else:
                roi_volume = imgs[mod_idx]
            save_name = "{0:}_{1:}.nii.gz".format(patient_dir, save_postfix[mod_idx])
            save_name = os.path.join(save_folder, save_name)
            save_array_as_nifty_volume(roi_volume, save_name)


if __name__ == "__main__":
#     get_training_set_statistics()
    # brats 15 crop and rename
    validation_data_source = '/Users/guotaiwang/Documents/data/BRATS/Brats2015_Training/HGG'
    validation_data_save = '/Users/guotaiwang/Documents/data/BRATS/Brats2015_Train_croprename/HGG'
    Brats15_data_set_crop_rename(validation_data_source,validation_data_save, True)
    # brats 17 validation crop and rename, for validation data, no crop
#     validation_data_source = '/Users/guotaiwang/Documents/data/BRATS2017/Brats17TestingData'
#     validation_data_save = '/Users/guotaiwang/Documents/data/BRATS2017/Brats17TestingData_renamed'
#     Brats17_data_set_crop_rename(validation_data_source,validation_data_save, False, False)    
#     
#     load_name = '/Users/guotaiwang/Documents/data/BRATS2017/Brats17TrainingData_crop_renamed/HGG/HGG1_FLAIR.nii.gz'
#     volume = load_nifty_volume_as_array(load_name)
#     print volume.shape
#     sub_volume = volume[0:100][:][:]
#     print sub_volume.shape
#     save_folder = '/Users/guotaiwang/Documents/workspace/tf_project/tf_brats/data_process/temp_data'
#     save_name = save_folder + '/Flair1sub.nii.gz'
#     save_array_as_nifty_volume(sub_volume, save_name)
    
