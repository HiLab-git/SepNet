# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import dicom
import numpy as np
from util.pre_function import get_segmented_body
from skimage import measure
from util.data_augament import ThreeDclip

def make_3d_groundtruth_only(patient_file_root, duplicate_path, label_wanted, label_aug, window_max, window_min, show_body = False, show_label = False):

    maskpoint = np.ones([512, 512, 2])  # 记录图像每点坐标
    patient_count = 0

    for x in range(512):
        for y in range(512):
            maskpoint[x, y] = [x, y]
    patient_number = len(os.listdir(patient_file_root))
    train_number = int(0.5*patient_number)
    test_number = int(0.7*patient_number)

    for patient_file in os.listdir(patient_file_root):

            zmax = 0    # 记录Label的最大最小z值
            zmin = 300
            patient_count += 1
            print('正在处理患者%s,这是第%d个病人' % (patient_file, patient_count))
            ctslices = []
            ctnumber = 0
            for s in os.listdir(str(patient_file_root) + '/' + patient_file):
                if 'CT' in s:
                    ctnumber += 1
                    ctslices.append(dicom.read_file(str(patient_file_root)+'/'+patient_file + '/' + s))
                if 'RS' in s:
                    rsslices = dicom.read_file(str(patient_file_root)+'/'+patient_file + '/' + s)    # 读入RS文件

            ctslices.sort(key=lambda x : int(x.ImagePositionPatient[2]))    # 按z坐标从小到大排序
            origin = [s.ImagePositionPatient for s in ctslices]    # 网格原点在世界坐标系的位置
            spacing = ctslices[0].PixelSpacing    # 采样间隔
            labeldata = np.zeros([ctnumber, 512, 512])
            imgdata = np.zeros([ctnumber, 512, 512])
            aug_label = np.zeros(ctnumber)
            body_box = np.zeros([ctnumber, 4])
            intercept = ctslices[0].RescaleIntercept    # 重采样截距
            slope = ctslices[0].RescaleSlope    # 重采样斜率

            '''
            提取患者身体区域
            '''
            znumber = 0
            for ct in ctslices:

                ctimg = np.array(ct.pixel_array)
                ctimg[ctimg == -2000] = 0
                if slope != 1:
                    ctimg = slope * ctimg.astype(np.float64)
                    ctimg = ctimg.astype(np.int64)
                ctimg = ctimg.astype(np.int64)
                ctimg += np.int64(intercept)

                body_img, body_mask = get_segmented_body(ctimg, window_max=window_max, window_min=window_min,
                                                         window_length=0,
                                                         show_body=show_body, znumber=znumber)
                labels = measure.label(body_mask)
                regions = measure.regionprops(labels)[0].bbox
                body_box[znumber] = regions
                imgdata[znumber] = body_img
                znumber += 1
            
            '''
            提取患者的靶区
            '''
            for i in range(len(rsslices.RTROIObservationsSequence)):    # 第i个勾画区域
                label = rsslices.RTROIObservationsSequence[i].ROIObservationLabel    # ROIObservationLabel即该ROI是何器官
                label = label.lower()
                label = label.replace(' ', '')
                label = label.replace('-', '')
                label = label.replace('_', '')

                if label in label_wanted:
                    print(label)
                    for j in range(len(rsslices.ROIContourSequence[i].ContourSequence)):    # 第j层靶区曲线

                        "提取靶区轮廓线坐标并转换为世界坐标"
                        numberOfPoints = rsslices.ROIContourSequence[i].ContourSequence[j].NumberofContourPoints    # 该层曲线上点数
                        Data = rsslices.ROIContourSequence[i].ContourSequence[j].ContourData
                        conData = np.zeros([numberOfPoints, 3])     # 存储靶区曲线各点的世界坐标
                        pointdata = np.zeros([numberOfPoints, 2])   # 存储靶区曲线各点的体素坐标
                        Z = Data[2]
                        znumber = round((Z-origin[0][2])/3)
                        if znumber > zmax:
                            zmax = znumber
                        elif znumber < zmin:
                            zmin = znumber
                        for jj in range(numberOfPoints):
                            ii = jj * 3
                            conData[jj, 0] = Data[ii    ]     # 轮廓世界坐标系
                            conData[jj, 1] = Data[ii + 1]
                            conData[jj, 2] = Data[ii + 2]
                            pointdata[jj, 0] = round((conData[jj, 0] - origin[0][0])/spacing[0])    # 轮廓X坐标
                            pointdata[jj, 1] = round((conData[jj, 1] - origin[0][1])/spacing[1])    # 轮廓Y坐标

                        "生成靶区mask"
                        pointdata = np.array(pointdata)
                        polyline = Path(pointdata, closed=True)    # 制成闭合的曲线
                        maskpoint_reshape = maskpoint.reshape(512*512, 2)
                        pointin = polyline.contains_points(maskpoint_reshape)
                        maskpoint_reshape = maskpoint_reshape[pointin, :]
                        for k in maskpoint_reshape:
                            labeldata[znumber, int(k[1]), int(k[0])] = label_wanted[label]
                        # if label in label_aug:  # 判断是否需要增强
                        #     aug_label[znumber] = 1
            imgdata = imgdata[zmin-5:zmax+5]
            labeldata = labeldata[zmin-5:zmax+5]
            body_box = body_box[zmin-5:zmax+5]
            if show_label:
                for i in range(len(labeldata)):
                    if i % 5 == 0:
                        f, plots = plt.subplots(1, 2, figsize=(60, 60))
                        plots[0].imshow(imgdata[i], cmap=plt.cm.bone)
                        plots[1].imshow(imgdata[i]*labeldata[i])
                        plt.show()

            if patient_count <= train_number:
                ThreeDclip(imgdata, labeldata, 'body', 'train', body_box, duplicate_path, patient_file,
                           multinumber=4, show_label=False)  # 训练集需要裁剪增强

            elif train_number < patient_count <= test_number:
                ThreeDclip(imgdata, labeldata, 'body', 'test', body_box, duplicate_path, patient_file, multinumber=4)

            else:

                ThreeDclip(imgdata, labeldata, 'body', 'valid', body_box, duplicate_path, patient_file, multinumber=4)

            print('已成功存储患者%s的数据' % patient_file)




    return


