import os
import numpy as np
import nibabel

def load_nifty_volume_as_array(filename):
    # input shape [W, H, D]
    # output shape [D, H, W]
    seg = nibabel.load(filename)
    data = seg.get_data()
    data = np.transpose(data, [2,1,0])
    return data

def save_array_as_nifty_volume(data, filename):
    # numpy data shape [D, H, W]
    # nifty image shape [W, H, W]
    data = np.transpose(data, [2,1,0])
    seg = nibabel.Nifti1Image(data, np.eye(4))
    nibabel.save(seg, filename)

patientroot = '/lyc/Head-Neck/MICCAI-19-StructSeg/HaN_OAR_center_crop/valid/' #预测结果-+
uncertaininterval = [0,0.4505,0.6365,0.6931,1.011]
data_name = ['enseg.nii.gz','uncertain.nii.gz', 'crop_label.nii.gz']
error_list = np.zeros([10, 5])
patient_number =0
for patient in os.listdir(patientroot):
    print('segname is ',patient)
    '''
    根据原label得到新label/seg的名称,像素间距与储存路径
    '''
    segname = data_name[0]
    labelname = data_name[2]
    uncertainname = data_name[1]
    
    segpath = os.path.join(patientroot, patient, segname)
    labelpath = os.path.join(patientroot,patient, labelname)
    uncertainpath = os.path.join(patientroot,patient, uncertainname)
    
    seg = load_nifty_volume_as_array(segpath)
    label = load_nifty_volume_as_array(labelpath)
    uncertainty = load_nifty_volume_as_array(uncertainpath)
    errormap = np.zeros_like(seg)
    errormap[np.where(label!=seg)]=1
    print(np.sum(errormap))
    for i in range(len(uncertaininterval)):
        uncertainmap = np.zeros_like(seg)
        uncertainty_index = np.where(uncertainty==uncertaininterval[i])
        uncertainmap[uncertainty_index]=1
        errorsum = np.sum(uncertainmap*errormap)
        error_list[patient_number, i] = errorsum/len(uncertainty_index[0])
    print(error_list)
    patient_number+=1
