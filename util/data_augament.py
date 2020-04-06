import numpy as np
import os
import matplotlib.pyplot as plt
import random
import cv2
from skimage import measure


def flip(img_path, save_path, show_img=False, prob=0.8):

    img_num = len(os.listdir(img_path))
    flipnum = 1
    for file in os.listdir(img_path):
        horizon_prob = random.random()
        vertical_prob = random.random()
        if horizon_prob < prob:
            img = np.load(img_path + '/' + file)
            img_horizon = img[:, :, ::-1]
            np.save(save_path + '/' + file[:-4] + '_horizon', img_horizon)

        if vertical_prob < prob:
            img = np.load(img_path + '/' + file)
            img_vertical = img[:, ::-1, :]
            np.save(save_path + '/' + file[:-4] + '_vertical', img_vertical)
        flipnum += 1

        if show_img and vertical_prob < prob and horizon_prob < prob:
            f, plots = plt.subplots(3, 2, figsize=[60, 60])
            plots[0, 0].axis('off')
            plots[0, 0].set_title('img')
            plots[0, 0].imshow(img[0], cmap=plt.cm.bone)
            plots[0, 1].axis('off')
            plots[0, 1].set_title('ground truth')
            plots[0, 1].imshow(img[1], cmap=plt.cm.bone)
            plots[1, 0].axis('off')
            plots[1, 0].set_title('img horizon flip')
            plots[1, 0].imshow(img_horizon[0], cmap=plt.cm.bone)
            plots[1, 1].axis('off')
            plots[1, 1].set_title('ground truth flip')
            plots[1, 1].imshow(img_horizon[1], cmap=plt.cm.bone)
            plots[2, 0].axis('off')
            plots[2, 0].set_title('img vertical flip')
            plots[2, 0].imshow(img_vertical[0], cmap=plt.cm.bone)
            plots[2, 1].axis('off')
            plots[2, 1].set_title('ground truth flip')
            plots[2, 1].imshow(img_vertical[1], cmap=plt.cm.bone)
            plt.show()
        print('翻转已完成 %s/%s' % (flipnum, img_num))

    return


def rotation(img_path, save_path, show_img=False, prob=0.8):

    img_num = len(os.listdir(img_path))
    rotatenum = 1
    for file in os.listdir(img_path):
        rotate_prob = random.random()
        if rotate_prob < prob:
            img = np.load(img_path + '/' + file)
            img_rotation_counter = rotate(img, 20)
            img_rotation = rotate(img, -20)
            # np.save(save_path + '/' + file[:-4] + '_rotation', img_rotation)
            # np.save(save_path + '/' + file[:-4] + '_counter_rotation', img_rotation_counter)
        rotatenum += 1
        if show_img and rotate_prob < prob:
            f, plots = plt.subplots(3, 2, figsize=[60, 60])
            plots[0, 0].axis('off')
            plots[0, 0].set_title('img')
            plots[0, 0].imshow(img[0], cmap=plt.cm.bone)
            plots[0, 1].axis('off')
            plots[0, 1].set_title('ground truth')
            plots[0, 1].imshow(img[1], cmap=plt.cm.bone)
            plots[1, 0].axis('off')
            plots[1, 0].set_title('img counter rotation')
            plots[1, 0].imshow(img_rotation_counter[0], cmap=plt.cm.bone)
            plots[1, 1].axis('off')
            plots[1, 1].set_title('ground counter rotation')
            plots[1, 1].imshow(img_rotation_counter[1], cmap=plt.cm.bone)
            plots[2, 0].axis('off')
            plots[2, 0].set_title('img rotation')
            plots[2, 0].imshow(img_rotation[0], cmap=plt.cm.bone)
            plots[2, 1].axis('off')
            plots[2, 1].set_title('ground rotation')
            plots[2, 1].imshow(img_rotation[1], cmap=plt.cm.bone)
            plt.show()

        print('旋转已完成 %s/%s' % (rotatenum, img_num))

    return

def translation(img_path, save_path, show_img=False, prob=0.8):
    """
    平移图像用
    :param img_path:
    :param save_path:
    :param show_img:
    :param prob: 小于就平移
    :return:
    """

    img_num = len(os.listdir(img_path))
    translatenum = 1
    for file in os.listdir(img_path):
        rotate_prob = random.random()
        if rotate_prob < prob:
            img = np.load(img_path + '/' + file)
            img_up = translate(img, 0, 10)
            img_down = translate(img, 0, -10)
            np.save(save_path + '/' + file[:-4] + '_up', img_up)
            np.save(save_path + '/' + file[:-4] + '_down', img_down)
            translatenum += 1
        if show_img and rotate_prob < prob:
            f, plots = plt.subplots(3, 2, figsize=[60, 60])
            plots[0, 0].axis('off')
            plots[0, 0].set_title('img')
            plots[0, 0].imshow(img[0], cmap=plt.cm.bone)
            plots[0, 1].axis('off')
            plots[0, 1].set_title('ground truth')
            plots[0, 1].imshow(img[1], cmap=plt.cm.bone)
            plots[1, 0].axis('off')
            plots[1, 0].set_title('img counter rotation')
            plots[1, 0].imshow(img_up[0], cmap=plt.cm.bone)
            plots[1, 1].axis('off')
            plots[1, 1].set_title('ground counter rotation')
            plots[1, 1].imshow(img_up[1], cmap=plt.cm.bone)
            plots[2, 0].axis('off')
            plots[2, 0].set_title('img rotation')
            plots[2, 0].imshow(img_down[0], cmap=plt.cm.bone)
            plots[2, 1].axis('off')
            plots[2, 1].set_title('ground rotation')
            plots[2, 1].imshow(img_down[1], cmap=plt.cm.bone)
            plt.show()

        print('平移已完成 %s/%s' % (translatenum, img_num))

    return

def rotate(img, angle, center=None, scale=1.0):
    '''
    旋转图像
    :param img:
    :param angle: 逆时针旋转角度
    :param center: 旋转中心,不指定默认为图像中心
    :param scale: 尺度变化参数
    :return:
    '''
    image = np.transpose(img, (1, 2, 0))
    (h, w) = image.shape[:2]
    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    rotated = np.transpose(rotated, (2, 0, 1))
    return rotated


def translate(img, x, y):
    '''
    图像平移函数
    原numpy文件通道数在第0维,cv2操作前需先转到第二维
    :param img: 原文件
    :param x: x轴平移距离
    :param y: y轴平移距离
    :return: 平移后图像
    '''

    image = np.transpose(img, (1, 2, 0))
    # 定义平移矩阵
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    shifted = np.transpose(shifted, (2, 0, 1))
    # 返回转换后的图像
    return shifted


def clip(img, module=None, clip_module='test', box=None, patient_name=None, znumber=None, width=256, length=256, multipl=1, multinumber=4):
    '''
    :param img:
    :param box: 指定身体区域,为行列最小与行列最大值
    :param module: 判断分割身体还是器官
    :param patient_name: 患者姓名
    :param znumber : 当前图像纵坐标
    :param width : 裁剪图像的宽
    :param length : 裁剪图像的高
    :param multipl: 判断是否要裁取多个图,即是否4倍增强.
    :param multinumber: 增强倍数
    :return: 裁剪后图像
    '''
    if module == 'body':
        center = [(box[0]+box[2])/2, (box[1]+box[3])/2]
        body_length = box[2] - box[0]
        body_width = box[3] - box[1]
        if body_width < width and body_length < length:
            if multipl == 1:
                height_random = min(box[0], length - body_length)
                width_random = min(box[1], width - body_width)
                for i in range(multinumber):
                    rowmin = int(box[0] - random.randint(0, height_random))  # 裁取图像的行坐标起点
                    colmin = int(box[1] - random.randint(0, width_random))  # 裁取图像的纵坐标起点
                    Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
                    plt.imshow(Img[0], cmap='bone')
                    plt.show()
                    np.save('/lyc/RTData/Parotid256/train/%s_%s_multi%s' % (patient_name, znumber, i), Img)
            else:
                rowmin = int(center[0] - width)
                colmin = int(center[1] - width)
                Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
                plt.imshow(img[0], cmap='bone')
                plt.show()
                np.save('/lyc/RTData/OpticChaism128/train/%s_%s' % (patient_name, znumber), Img)

    elif module == 'organism':
        labels = measure.label(img[1])
        regions = measure.regionprops(labels)
        if len(regions) > 0:
            box = [512, 512, 0, 0]
            for labelnumber in regions:  # 找出该ct中所有标注的最大最小横纵坐标值.
                box[0], box[1] = min(labelnumber.bbox[0], box[0]), min(labelnumber.bbox[1], box[1])
                box[2], box[3] = max(labelnumber.bbox[2], box[2]), max(labelnumber.bbox[3], box[3])

            # 为了避免裁剪时让label太靠近边界,故框四边各向外扩张16
            box[0] -= 16
            box[1] -= 16
            box[2] += 16
            box[3] += 16

            label_height = box[2] - box[0]
            label_width = box[3] - box[1]
            if label_width < width and label_height < length:
                if multipl == 1:
                    row_random_max = min(box[0], 512 - width)
                    row_random_min = max(box[2]-width, 0)
                    col_random_max = min(box[1], 512 - length)
                    col_random_min = max(box[3]-length, 0)
                    for i in range(multinumber):
                        colmin = random.randint(col_random_min, col_random_max)  # 裁取图像的行坐标起点
                        rowmin = random.randint(row_random_min, row_random_max)  # 裁取图像的纵坐标起点
                        Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
                        # f, plots = plt.subplots(1, 4, figsize=(60, 60))
                        # plots[0].imshow(Img[0], cmap='bone')
                        # plots[1].imshow(Img[1])
                        # plots[2].imshow(Img[1]*Img[0])
                        # plots[3].imshow(img[0])
                        # plt.show()
                        np.save('/lyc/RTData/OpticNerve/%s/%s/%s_%s_multi%s' % (width, clip_module, patient_name, znumber, i), Img )

    return

def mkdir(path):
    """
    创建path所给文件夹
    :param path:
    :return:
    """
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("---  new folder...  ---")

        print("---  OK  ---")

    else:
        print("---  There is this folder!  ---")


def ThreeDclip(img, label, module=None, clip_module='test', box=None, duplicate_path=None,
               patient_name=None, znumber=None, width=256, length=256, multipl=0, multinumber=4, show_label=False):
    '''
    :param img:
    :param box: 指定身体区域,为行列最小与行列最大值
    :param module: 判断分割身体还是器官
    :param patient_name: 患者姓名
    :param znumber : 当前图像纵坐标
    :param width : 裁剪图像的宽
    :param length : 裁剪图像的高
    :param multipl: 判断是否要裁取多个图,即是否4倍增强.
    :param multinumber: 增强倍数
    :return: 裁剪后图像
    '''
    save_path = duplicate_path + '/' + clip_module + '/' + patient_name
    mkdir(save_path)
    if module == 'body':
        body_length = max(box[:, 2]) - min(box[:, 0])
        body_width = max(box[:, 3]) - min(box[:, 1])
        if body_width > width or body_length > length:
            print('图像设置太小, patient: %s, 所需宽: %d, 所需长: %d' % (patient_name, body_width, body_length))
            body_length = length
            body_width = width
            center = [256, 256]
        else:
            center = [(min(box[:, 0]) + max(box[:, 2])) / 2, (min(box[:, 1]) + max(box[:, 3])) / 2]
        if multipl == 1:
            height_random = min(box[0], length - body_length)
            width_random = min(box[1], width - body_width)
            for i in range(multinumber):
                rowmin = int(box[0] - random.randint(0, height_random))  # 裁取图像的行坐标起点
                colmin = int(box[1] - random.randint(0, width_random))  # 裁取图像的纵坐标起点
                Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
                label = label[:, rowmin:rowmin + width, colmin:colmin + length]
                if show_label:
                    for i in range(len(label)):
                        if i % 5 == 0:
                            f, plots = plt.subplots(1, 2, figsize=(60, 60))
                            plots[0].imshow(Img[i], cmap=plt.cm.bone)
                            plots[1].imshow(Img[i] * label[i])
                            plt.show()
                np.save(save_path + '/' + 'Img.npy', Img)
                np.save(save_path + '/' + 'label.npy', label)
        else:
            rowmin = int(center[0] - width/2)
            colmin = int(center[1] - length/2)
            Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
            label = label[:, rowmin:rowmin + width, colmin:colmin + length]
            if show_label:
                for i in range(len(label)):
                    if i % 5 == 0:
                        f, plots = plt.subplots(1, 2, figsize=(60, 60))
                        plots[0].imshow(Img[i], cmap=plt.cm.bone)
                        plots[1].imshow(Img[i] * label[i])
                        plt.show()
            np.save(save_path + '/' + 'Img.npy', Img)
            np.save(save_path + '/' + 'label.npy', label)


    elif module == 'organism':
        labels = measure.label(img[1])
        regions = measure.regionprops(labels)
        if len(regions) > 0:
            box = [512, 512, 0, 0]
            for labelnumber in regions:  # 找出该ct中所有标注的最大最小横纵坐标值.
                box[0], box[1] = min(labelnumber.bbox[0], box[0]), min(labelnumber.bbox[1], box[1])
                box[2], box[3] = max(labelnumber.bbox[2], box[2]), max(labelnumber.bbox[3], box[3])

            # 为了避免裁剪时让label太靠近边界,故框四边各向外扩张16
            box[0] -= 16
            box[1] -= 16
            box[2] += 16
            box[3] += 16

            label_height = box[2] - box[0]
            label_width = box[3] - box[1]
            if label_width < width and label_height < length:
                if multipl == 1:
                    row_random_max = min(box[0], 512 - width)
                    row_random_min = max(box[2]-width, 0)
                    col_random_max = min(box[1], 512 - length)
                    col_random_min = max(box[3]-length, 0)
                    for i in range(multinumber):
                        colmin = random.randint(col_random_min, col_random_max)  # 裁取图像的行坐标起点
                        rowmin = random.randint(row_random_min, row_random_max)  # 裁取图像的纵坐标起点
                        Img = img[:, rowmin:rowmin + width, colmin:colmin + length]
                        # f, plots = plt.subplots(1, 4, figsize=(60, 60))
                        # plots[0].imshow(Img[0], cmap='bone')
                        # plots[1].imshow(Img[1])
                        # plots[2].imshow(Img[1]*Img[0])
                        # plots[3].imshow(img[0])
                        # plt.show()
                        np.save('/lyc/RTData/OpticNerve/%s/%s/%s_%s_multi%s' % (width, clip_module, patient_name, znumber, i), Img )

    return

if __name__ == "__main__":
    img_path = '/lyc/RTData/OpticNerve/256/train'
    save_path = '/lyc/RTData/OpticNerve/256/aug'
    show_img = False

    flip(img_path, save_path,  show_img=show_img, prob=0.8)
    rotation(img_path, save_path, show_img=show_img, prob=0.8)
    # translation(img_path, save_path, show_img=show_img, prob=0.8)