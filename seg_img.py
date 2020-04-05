#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import time
import os
from torch.nn import parallel
import torch.tensor
from models.LYC_data_loader import LYC_dataset
from util.train_test_func import *
from util.parse_config import parse_config
from models.Unet_Separate_2 import Unet_Separate_2
from models.Unet_Separate_3 import Unet_Separate_3
from models.Unet_Separate_4 import Unet_Separate_4
from models.UnetSE_Separate_3 import UnetSE_Separate_3
from models.UnetDense_Separate_3 import UnetDense_Separate_3
from models.UnetDense_Separate_4 import UnetDense_Separate_4
from models.UnetDense_Separate_5 import UnetDense_Separate_5
from models.Unet import Unet
from models.Unet_Res import Unet_Res
from models.SOLNet import SOLNet
from models.Unet_Separate import Unet_Separate
from models.DS_Unet_Separate_3 import DS_Unet_Separate_3
from models.DS_Unet_Separate_4 import DS_Unet_Separate_4
from models.loss_function import TestDiceLoss
from util.binary import assd, dc, hd95
from data_process.data_process_func import save_array_as_nifty_volume, make_overlap_weight
from util.assd_evaluation import one_hot
from skimage import morphology
import pandas as pd

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'Unet_Separate_2':
            return Unet_Separate_2

        if name == 'Unet':
            return Unet

        if name == 'Unet_Res':
            return Unet_Res

        if name == 'Unet_Separate':
            return Unet_Separate

        if name == 'Unet_Separate_3':
            return Unet_Separate_3

        if name == 'Unet_Separate_4':
            return Unet_Separate_4

        if name == 'UnetSE_Separate_3':
            return UnetSE_Separate_3

        if name == 'SOLNet':
            return SOLNet

        if name == 'UnetDense_Separate_3':
            return UnetDense_Separate_3

        if name == 'UnetDense_Separate_4':
            return UnetDense_Separate_4

        if name == 'UnetDense_Separate_5':
            return UnetDense_Separate_5

        if name == 'DS_Unet_Separate_3':
            return DS_Unet_Separate_3

        if name == 'DS_Unet_Separate_4':
            return DS_Unet_Separate_4

        # add your own networks here
        print('unsupported network:', name)
        exit()


def seg(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # config of data,e.g. data_shape,batch_size.
    config_net = config['network']  # config of net, e.g. net_name,base_feature_name,class_num.
    config_test = config['testing']
    overlap_num = config['data']['overlap_num']
    random.seed(config_test.get('random_seed', 1))
    subseg_name = config_data['seg_name']
    subprob_name = subseg_name.replace('seg', 'prob')
    net_type = config_net['net_type']
    class_num = config_net['class_num']
    overlap_weight = make_overlap_weight(overlap_num)
    output_probability = False
    save = False
    save_array_as_xls = False
    show = False
    cal_dice = True
    cal_assd = False
    cal_hd95 = False
    show_hist = False
    overlap_bias = False
    class_weight = np.asarray([0, 100,100,50,50,80,80,50,80,80,80,50,50,70,70,70,70,60,60,100,100,100])


    # 2, load data
    print('2.Load data')
    Datamode = ['valid']

    # 3. creat model
    print('3.Creat model')
    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes = class_num,
                    base_chns= config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=False,
                    dilation=config_net.get('dilation', 1)
                    )

    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    if config_test['load_weight']:
        weight = torch.load(config_test['model_path'], map_location=lambda storage, loc: storage)
        net.load_state_dict(weight)
    print(torch.cuda.is_available())


    # 4, start to seg
    print('''start to seg ''')
    net.eval()
    for mode in Datamode:
        Data = LYC_dataset(config_data, mode)
        patient_number = len(os.listdir(os.path.join(config_data['data_root'], mode)))
        with torch.no_grad():
            t_array = np.zeros(patient_number)
            dice_array = np.zeros([patient_number, class_num])
            assd_array = np.zeros([patient_number, class_num])
            hd95_array = np.zeros([patient_number, class_num])
            for patient_order in range(patient_number):
                t1 = time.time()
                valid_pair, patient_path = Data.get_list_overlap_img(patient_order)  # 因为病人的数据无法一次完全预测,内存不够,所以裁剪成几块
                clip_number = len(valid_pair['images'])  # 裁剪块数
                clip_height = config_data['test_data_shape'][0]  # 裁剪图像的高度
                total_labels = valid_pair['labels'].cuda()
                predic_size = torch.Size([1, class_num]) + total_labels.size()[1::]
                totalpredic = torch.zeros(predic_size).cuda()  # 完整预测
                outfeature_size = torch.Size([1, 2*config_net.get('base_feature_number')]) + total_labels.size()[1::]
                totalfeature = torch.zeros(outfeature_size).cuda()
                for i in range(clip_number):
                    tempx = valid_pair['images'][i].cuda()
                    pred = net(tempx)
                    if overlap_bias:
                        for j in range(overlap_num):
                            pred[:,:,j*clip_height:(j+1)*clip_height]*= overlap_weight[j]
                    if i < clip_number - 1:
                        totalpredic[:, :, i * clip_height:(i + overlap_num) * clip_height] += pred
                    else:
                        totalpredic[:, :, -overlap_num*clip_height::] += pred

                if output_probability:
                    totalfeature = (100*totalpredic.cpu().data.numpy().squeeze()).astype(np.uint16)
                totalpredic = torch.max(totalpredic, 1)[1].squeeze()
                totalpredic = np.uint8(totalpredic.cpu().data.numpy().squeeze())
                totallabel = np.uint8(total_labels.cpu().data.numpy().squeeze())

                t2 = time.time()
                t = t2-t1
                t_array[patient_order] = t

                one_hot_label = one_hot(totallabel, class_num)
                one_hot_predic = one_hot(totalpredic, class_num)
                # for i in range(one_hot_predic[20].shape[0]):
                #     one_hot_predic[20, i] = morphology.erosion(one_hot_predic[20, i], np.ones([1, 1]))

                if cal_dice:
                    Dice = np.zeros(class_num)
                    for i in range(class_num):
                        Dice[i] = dc(one_hot_predic[i], one_hot_label[i])
                    dice_array[patient_order] = Dice
                    print('patient order', patient_order, ' dice:', Dice)

                if cal_assd:
                    Assd = np.zeros(class_num)
                    for i in range(class_num):
                        Assd[i] = assd(one_hot_predic[i], one_hot_label[i], 1)
                    assd_array[patient_order] = Assd
                    print('patient order', patient_order, ' dice:', Assd)

                if cal_hd95:
                    Hd95 = np.zeros(class_num)
                    for i in range(class_num):
                        Hd95[i] = hd95(one_hot_predic[i], one_hot_label[i], 1)
                    hd95_array[patient_order] = Hd95
                    print('patient order', patient_order, ' Hd95:', Hd95)


                if show:
                    for i in np.arange(0, totalpredic.shape[0], 2):
                        f, plots = plt.subplots(1, 2)
                        plots[0].imshow(totalpredic[i])
                        plots[1].imshow(totallabel[i])
                        plt.show()
                if save :
                    if output_probability:
                        save_array_as_nifty_volume(totalfeature, patient_path+'/'+subprob_name, transpose=False, pixel_spacing=[1,1,1,1])
                        # np.save(patient_path+'/'+subseg_name, totalfeature)
                    save_array_as_nifty_volume(totalpredic, patient_path + '/' +subseg_name)
                    # np.savetxt(patient_path+'/Dice.npy', Dice.squeeze())
                    # np.savetxt(patient_path+'/Assd.npy', Assd.squeeze())

        if cal_dice:
            dice_array[:, 0] = np.mean(dice_array[:, 1::], 1)
            dice_mean = np.mean(dice_array, 0)
            dice_std = np.std(dice_array, 0)
            # weight_score = np.inner(dice_mean, class_weight)
            print('{0:} mode: mean dice:{1:}, std of dice:{2:}'.format(mode, dice_mean, dice_std))#, weight_score))
            if show_hist:
                plt.figure('hist')
                for i in range(class_num-1):
                    plt.subplot(4, 6, i+1)
                    plt.hist(dice_array[:, i+1], bins=10, range=(0, 1))
                plt.show()
        if cal_assd:
            assd_array[:, 0] = np.mean(assd_array[:, 1::], 1)
            assd_mean = np.mean(assd_array, 0)
            assd_std = np.std(assd_array, 0)
            print('{0:} mode: mean assd:{1:}, std of assd:{2:}'.format(mode, assd_mean, assd_std))

        if cal_hd95:
            hd95_array[:, 0] = np.mean(hd95_array[:, 1::], 1)
            hd95_mean = np.mean(hd95_array, 0)
            hd95_std = np.std(hd95_array, 0)
            if save_array_as_xls:
                mean = pd.DataFrame(hd95_mean)
                mean_writer = pd.ExcelWriter('hd95_mean.xlsx')
                mean.to_excel(mean_writer, 'page_1', float_format='%.3f')
                mean_writer.save()
                mean_writer.close()

                std = pd.DataFrame(hd95_std)
                std_writer = pd.ExcelWriter('hd95_std.xlsx')
                std.to_excel(std_writer, 'page_1', float_format='%.3f')
                std_writer.save()
                std_writer.close()
            print('{0:} mode: mean HD95:{1:}, std of HD95:{2:}'.format(mode, hd95_mean, hd95_std))
        t_mean = [t_array.mean()]
        t_std = [t_array.std()]
        print('{0:} mode: mean time:{1:}, std of time:{2:}'.format(mode, t_mean, t_std))



if __name__ == '__main__':
    #for i in range(6):
        config_file = str('config/pnet_test.txt')
        assert (os.path.isfile(config_file))
        seg(config_file)
