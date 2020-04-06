# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import time
from torch.nn import parallel
import torch.tensor
from models.LYC_data_loader import LYC_dataset
from util.train_test_func import *
from util.parse_config import parse_config
from models.Unet import Unet
from models.Unet_Separate import Unet_Separate
from util.binary import assd, dc
from data_process.data_process_func import save_array_as_nifty_volume
from util.assd_evaluation import one_hot

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'Unet':
            return Unet

        if name == 'Unet_Separate':
            return Unet_Separate

        # add your own networks here
        print('unsupported network:', name)
        exit()


def seg(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data = config['data']  # config of data,e.g. data_shape,batch_size.
    config_net = config['network']  # config of net, e.g. net_name,base_feature_name,class_num.
    config_train = config['training']
    random.seed(config_train.get('random_seed', 1))
    output_feature = config_data.get('output_feature', False)
    net_type = config_net['net_type']
    class_num = config_net['class_num']
    save = False
    show = False
    cal_dice = True
    cal_assd = False

    # 2, load data
    print('2.Load data')
    Datamode = ['valid']

    # 3. creat model
    print('3.Creat model')
    # dice_eval = TestDiceLoss(class_num)
    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes = class_num,
                    base_chns= config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=False,
                    dilation=1
                    )

    net = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()
    if config_train['load_weight']:
        weight = torch.load(config_train['model_path'], map_location=lambda storage, loc: storage)
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
            for patient_order in range(patient_number):
                t1 = time.time()
                valid_pair, patient_path = Data.get_list_img(patient_order)  
                clip_number = len(valid_pair['images'])  # 裁剪块数
                clip_height = config_data['test_data_shape'][0]  # 裁剪图像的高度
                total_labels = valid_pair['labels'].cuda()
                predic_size = torch.Size([1, class_num]) + total_labels.size()[1::]
                totalpredic = torch.zeros(predic_size).cuda()  # 完整预测
                if output_feature:
                    outfeature_size = torch.Size([1, 2*config_net.get('base_feature_number')]) + total_labels.size()[1::]
                    totalfeature = torch.zeros(outfeature_size).cuda()
                for i in range(clip_number):
                    tempx = valid_pair['images'][i].cuda()
                    if output_feature:
                        pred, outfeature = net(tempx)
                    else:
                        pred = net(tempx)
                    if i < clip_number - 1:
                        totalpredic[:, :, i * clip_height:(i + 1) * clip_height] = pred
                    else:
                        totalpredic[:, :, -clip_height::] = pred
                    if output_feature:
                        if i < clip_number - 1:
                            totalfeature[:, :, i * clip_height:(i + 1) * clip_height] = outfeature
                        else:
                            totalfeature[:, :, -clip_height::] = outfeature

                # torchdice = dice_eval(totalpredic, total_labels)
                # print('torch dice:', torchdice)
                totalpredic = torch.max(totalpredic, 1)[1].squeeze()
                totalpredic = np.uint8(totalpredic.cpu().data.numpy().squeeze())
                totallabel = np.uint8(total_labels.cpu().data.numpy().squeeze())
                if output_feature:
                    totalfeature = totalpredic.cpu().data.numpy().squeeze()
                t2 = time.time()
                t = t2-t1
                t_array[patient_order] = t

                one_hot_label = one_hot(totallabel, class_num)
                one_hot_predic = one_hot(totalpredic, class_num)

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

                if show:
                    for i in np.arange(0, totalpredic.shape[0], 2):
                        f, plots = plt.subplots(1, 2)
                        plots[0].imshow(totalpredic[i])
                        plots[1].imshow(totallabel[i])
                        #plots[2].imshow(oriseg[i])
                        # plots[1, 0].imshow(totalfeature[0, i])
                        # plots[1, 1].imshow(totalfeature[5, i])
                        plt.show()
                if save :
                    if output_feature:
                        np.save(patient_path + '/Feature.npy', totalfeature)
                    #np.save(patient_path + '/Seg_2.npy', totalpredic)
                    save_array_as_nifty_volume(totalpredic, patient_path + '/Seg.nii.gz')
                    # np.savetxt(patient_path+'/Dice.npy', Dice.squeeze())
                    # np.savetxt(patient_path+'/Assd.npy', Assd.squeeze())

        if cal_dice:
            dice_array[:, 0] = np.mean(dice_array[:, 1::], 1)
            dice_mean = np.mean(dice_array, 0)
            dice_std = np.std(dice_array, 0)
            print('{0:} mode: mean dice:{1:}, std of dice:{2:}'.format(mode, dice_mean, dice_std))

        if cal_assd:
            assd_array[:, 0] = np.mean(assd_array[:, 1::], 1)
            assd_mean = np.mean(assd_array, 0)
            assd_std = np.std(assd_array, 0)
            print('{0:} mode: mean assd:{1:}, std of assd:{2:}'.format(mode, assd_mean, assd_std))

        t_mean = [t_array.mean()]
        t_std = [t_array.std()]
        print('{0:} mode: mean time:{1:}, std of time:{2:}'.format(mode, t_mean, t_std))

config_file = str('config/pnet_train.txt')
assert (os.path.isfile(config_file))
seg(config_file)
