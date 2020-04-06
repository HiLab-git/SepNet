# -*- coding:utf-8 -*-

import os
import matplotlib.pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import dicom
import numpy as np

def collect_organism_hist(path,Label_wanted):

    Windows_img = {}   # 用来保存窗口图片
    maskpoint = np.ones([512, 512, 2])  # 生成各点坐标
    patient_count = 0
    for x in range(512):
        for y in range(512):
            maskpoint[x, y] = [x, y]

    for patient_file in os.listdir(path):
        try:
            patient_count += 1
            print('正在处理患者%s,这是第%d个病人' % (patient_file, patient_count))
            ctslices = []
            ctnumber = 0    # 记录CT个数
            for s in os.listdir(str(path) + '/' + patient_file):    # 载入文件
                if 'CT' in s:
                    ctnumber += 1
                    ctslices.append(dicom.read_file(str(path)+'/'+patient_file + '/' + s))    # 判断是否为CT
                if 'RS' in s:
                    rsslices = dicom.read_file(str(path)+'/'+patient_file + '/' + s)    # 读入RS文件

            ctslices.sort(key=lambda x: int(x.ImagePositionPatient[2]))    # 按z坐标从小到大排序
            origin = [s.ImagePositionPatient for s in ctslices]    # 网格原点在世界坐标系的位置
            spacing = ctslices[0].PixelSpacing    # 采样间隔
            intercept = ctslices[0].RescaleIntercept    # 重采样截距
            slope = ctslices[0].RescaleSlope    # 重采样斜率


            "提取患者的第I个靶区"
            for i in range(len(rsslices.RTROIObservationsSequence)):    # 第i个勾画区域
                label = rsslices.RTROIObservationsSequence[i].ROIObservationLabel    # ROIObservationLabel即该ROI是何器官
                label = label.lower()
                label = label.replace(' ', '')
                label = label.replace('-', '')
                label = label.replace('_', '')

                if label in Label_wanted:
                    try:
                        print(label)
                        maskdata = np.zeros([ctnumber, 512, 512])
                        for j in range(len(rsslices.ROIContourSequence[i].ContourSequence)):    # 第j层靶区曲线

                            "提取靶区轮廓线坐标并转换为世界坐标"
                            numberOfPoints = rsslices.ROIContourSequence[i].ContourSequence[j].NumberofContourPoints    # 该层曲线上点数
                            Data = rsslices.ROIContourSequence[i].ContourSequence[j].ContourData
                            conData = np.zeros([numberOfPoints, 3])     # 存储靶区曲线各点的世界坐标
                            pointdata = np.zeros([numberOfPoints, 2])   # 存储靶区曲线各点的体素坐标
                            Z = Data[2]                     # 当前曲线的z坐标
                            znumber = round((Z-origin[0][2])/3)       # 当前靶线是第几层CT
                            ctimg = np.array(ctslices[round(znumber)].pixel_array)
                            ctimg[ctimg == -2000] = 0
                            if slope != 1:
                                ctimg = slope * ctimg.astype(np.float64)
                                ctimg = ctimg.astype(np.int16)
                            ctimg = ctimg.astype(np.int16)
                            ctimg += np.int16(intercept)
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
                                maskdata[znumber][int(k[0]), int(k[1])] = 1
                            if label in Windows_img:
                                Label_img = (maskdata[znumber] * ctimg).flatten()
                                Label_cor = Label_img != 0
                                Label_img = Label_img[Label_cor]
                                np.concatenate((Windows_img[Label_wanted[label]], Label_img))
                            else:
                                Label_img = (maskdata[znumber]*ctimg).flatten()
                                Label_cor = Label_img != 0
                                Label_img = Label_img[Label_cor]
                                Windows_img[Label_wanted[label]] = Label_img

                    except:
                            print('患者%s的%s标注有点问题' % (patient_file, label))
        except:
            print('患者%s的图像数据有些问题' % patient_file)

    # Windows_img[Label_wanted][Windows_img[Label_wanted] > 200 ] = 0
    # Windows_img[Label_wanted][Windows_img[Label_wanted] < -200] = 0
    for label in Windows_img:

        plt.hist(Windows_img[label], log=True)
        plt.title('%s histogram' % label)
        plt.show()

    return


if __name__ == "__main__":

    path = '/lyc/RTData/Original CT/RT'
    Label_wanted = label_wanted
    collect_organism_hist(path, Label_wanted)