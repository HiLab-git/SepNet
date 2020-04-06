
# -*- coding:utf-8 -*-


'''
该文件主要包含图像处理的函数
'''
import os
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.filters import roberts
from skimage import measure
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import dicom
import scipy.misc
import numpy as np


# Load the scans in given folder path
def load_scan(path):
    '''
     该函数用于载入path下各患者的CT,并从中提取图像储存在各自的文件夹
    '''
    Patient_count = 0
    for Patient_file in os.listdir(path):
        try:
            Patient_count += 1
            CTslices = []
            CTnumber = 0  # 记录CT个数
            print('正在处理患者%s,这是第%d个病人' % (Patient_file, Patient_count))
            for s in os.listdir(str(path) + '/' + Patient_file):  # 载入文件
                if 'CT' in s:
                    CTnumber += 1
                    CTslices.append(dicom.read_file(str(path) + '/' + Patient_file + '/' + s))  # 判断是否为CT
            CTslices.sort(key=lambda x: int(x.ImagePositionPatient[2]))

            "提取CT间隔"
            print(CTslices[1].ImagePositionPatient[2], CTslices[2].ImagePositionPatient[2])
            try:
                slice_thickness = np.abs(CTslices[0].ImagePositionPatient[2] - CTslices[1].ImagePositionPatient[2])
            except:
                slice_thickness = np.abs(CTslices[0].SliceLocation - CTslices[1].SliceLocation)
            for s in CTslices:
                s.SliceThickness = slice_thickness

            CTimage = get_pixels_hu(CTslices)
            np.save(str(path) + '/' + Patient_file + '/Patient_pixel.npy', CTimage)
        except:
            print('提取图像中,患者%sCT图像有问题' % Patient_file)
    print('图像提取完成!')
    return


def get_pixels_hu(slices):
    # 灰度值转换为HU单元

    image = np.stack([s.pixel_array for s in slices])

    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.float64)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        # 回到HU单元，乘以rescale比率并加上intercept(存储在扫描面的元数据中)
        intercept = slices[slice_number].RescaleIntercept  # 截距
        slope = slices[slice_number].RescaleSlope  # 斜率
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.float64)
        image[slice_number] += np.float64(intercept)
    return np.array(image, dtype=np.float64)


def resample(image, scan, new_spacing=[1, 1, 1]):
    # 重采样
    # 不同扫描面的像素尺寸、粗细粒度是不同的。这不利于我们进行CNN任务，我们可以使用同构采样。
    # Determine current pixel spacing
    print('扫描面厚度,像素间距', [scan[0].SliceThickness] + scan[0].PixelSpacing)
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))
    resize_factor = spacing / new_spacing  # ？？？
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    print('new real shape is', new_real_shape, 'resize factor is', resize_factor)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # 插值法上采样
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


def plot_3d(image, threshold=-300):
    # 使用matplotlib输出肺部扫描的3D图像方法。可能需要一两分钟
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    verts, faces, x, y = measure.marching_cubes(p, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()


def plot_ct_scan(scan):
    # 输出一个病人scans中所有切面slices
    '''
            plot a few more images of the slices
    :param scan:
    :return:
    '''
    f, plots = plt.subplots(int(scan.shape[0] / 20), 4, figsize=(50, 50))
    for i in range(0, scan.shape[0], 5):
        plots[int(i / 20), int((i % 20) / 5)].axis('on')
        plots[int(i / 20), int((i % 20) / 5)].imshow(scan[i], cmap=plt.cm.bone)
    plt.show()


def get_segmented_body(img, window_max=250, window_min=-150, window_length=0, show_body=False, znumber=0):
    '''
    将身体与外部分离出来
    '''

    mask = []

    if znumber < 40:
        radius = [13, 6]
    else:
        radius = [6, 8]

    plot = False
    show_now = False
    if show_body:
        if znumber % 10 == 0:
            plot = True
            show_now = True

    if plot == True:
        f, plots = plt.subplots(2, 4, figsize=(60, 60))

    '''
    Step 1: Convert into a binary image.二值化,为确保所定阈值通过大多数
    '''
    threshold = -600
    binary = np.where(img > threshold, 1.0, 0.0)  # threshold the image

    if plot == True:
        plots[0, 0].axis('off')
        plots[0, 0].set_title('convert into a binary image,the the threshold%s' % threshold)
        plots[0, 0].imshow(binary, cmap=plt.cm.bone)
    '''
    Step 2: Remove the blobs connected to the border of the image.
            清除边界
    '''
    # cleared = clear_border(binary,buffer_size=50)
    # if plot == True:
    #     plots[0,1].axis('off')
    #     plots[0,1].set_title('after clear border')
    #     plots[0,1].imshow(cleared[0], cmap=plt.cm.bone)
    #     print(cleared[0])

    '''
    Step 3: Erosion operation with a disk of radius 2. This operation is
    seperate the lung nodules attached to the blood vessels.
    腐蚀操作，以2mm为半径去除
    '''
    binary = morphology.erosion(binary, np.ones([radius[0], radius[0]]))
    if plot == True:
        plots[0, 1].axis('off')
        plots[0, 1].set_title('erosion operation')
        plots[0, 1].imshow(binary, cmap=plt.cm.bone)

    '''
    Step 4: Closure operation with a disk of radius 10. This operation is
    to keep nodules attached to the lung wall.闭合运算
    '''
    binary = morphology.dilation(binary, np.ones([radius[1], radius[1]]))
    if plot == True:
        plots[0, 2].axis('off')
        plots[0, 2].set_title('closure operation')
        plots[0, 2].imshow(binary, cmap=plt.cm.bone)

    '''
    Step 5: Label the image.连通区域标记
    '''
    label_image = label(binary)
    if plot == True:
        plots[0, 3].axis('off')
        plots[0, 3].set_title('found all connective graph')
        plots[0, 3].imshow(label_image)

    '''
    Step 6: Keep the labels with the largest area.保留最大区域
    '''
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 1:
        for region in regionprops(label_image):
            if region.area < areas[-1]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    if plot == True:
        plots[1, 0].axis('off')
        plots[1, 0].set_title('keep the largest area')
        plots[1, 0].imshow(binary, cmap=plt.cm.bone)

    '''
    Step 7: Fill in the small holes inside the binary mask .孔洞填充
    '''
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    if plot == True:
        plots[1, 1].axis('off')
        plots[1, 1].set_title('fill in the small holes')
        plots[1, 1].imshow(binary, cmap=plt.cm.bone)

    '''
    Step 8: show the input image.
    '''
    if plot == True:
        plots[1, 3].axis('off')
        plots[1, 3].set_title('input image')
        plots[1, 3].imshow(img, cmap='gray')

    '''
    Step 9: Superimpose the binary mask on the input image.
    '''
    get_high_vals = binary == 0
    img[get_high_vals] = 0
    if plot == True:
        plots[1, 2].axis('off')
        plots[1, 2].set_title('superimpose the binary mask')
        plots[1, 2].imshow(img, cmap='gray')
    if show_now == True:
        plt.show()
    mask.append(binary)

    img[img > (window_max + window_length)] = window_max + window_length
    img[img < (window_min - window_length)] = window_min - window_length
    img = (img - window_min) / (window_max - window_min)
    img[get_high_vals] = 0
    if plot == True:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        plt.show()

    return img, binary


def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)
    counts = counts[vals != bg]
    vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None


def segment_lung_mask(image, fill_lung_structures=True):
    '''肺部图像分割
       为了减少有问题的空间，我们可以分割肺部图像（有时候是附近的组织）
       这包含一些步骤，包括区域增长和形态运算，此时，我们只分析相连组件
    '''

    # 1是空气，2是肺部
    # not actually binary, but 1 and 2.
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8) + 1
    labels = measure.label(binary_image)  # 连通区域标记

    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air
    #   around the person in half
    background_label = labels[0, 0, 0]
    # Fill the air around the person
    binary_image[background_label == labels] = 2
    # Method of filling the lung structures (that is superior to something like
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            if l_max is not None:  # This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    binary_image -= 1  # Make the image actual binary
    binary_image = 1 - binary_image  # Invert it, lungs are now 1
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None:  # There are air pockets
        binary_image[labels != l_max] = 0
    return binary_image
