#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import time
import torch.optim as optim
from torch.nn import parallel
import torch.tensor
from models.LYC_data_loader import LYC_dataset, set_noclass_zero
from util.train_test_func import *
from util.parse_config import parse_config
from models.Unet_Separate import Unet_Separate
from models.Unet import Unet
from models.loss_function import TestDiceLoss, SoftDiceLoss, ExpDiceLoss, AttentionExpDiceLoss, DiceLoss, make_one_hot
from models.Pymic_loss import soft_dice_loss, get_soft_label
from util.visualization.visualize_loss import loss_visualize
from util.visualization.show_param import show_param

class NetFactory(object):
    @staticmethod
    def create(name):
        if name == 'Unet':
            return Unet

        if name == 'Unet_Separate':
            return Unet_Separate

        print('unsupported network:', name)
        exit()

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def train(config_file):
    # 1, load configuration parameters
    print('1.Load parameters')
    config = parse_config(config_file)
    config_data  = config['data']    # data config, like data_shape,batch_size,
    config_net   = config['network']    # net config, like net_name,base_feature_name,class_num
    config_train = config['training']

    random.seed(config_train.get('random_seed', 1))     

    valid_patient_number = len(os.listdir(config_data['data_root']+'/'+'valid'))
    net_type    = config_net['net_type']
    class_num   = config_net['class_num']
    batch_size  = config_data.get('batch_size', 4)
    lr = config_train.get('learning_rate', 1e-3)
    best_dice = config_train.get('best_dice', 0.5)

    # 2, load data
    print('2.Load data')
    trainData = LYC_dataset(config_data, 'train')
    validData = LYC_dataset(config_data, 'valid')

    # 3. creat model
    print('3.Creat model')
    net_class = NetFactory.create(net_type)
    net = net_class(inc=config_net.get('input_channel', 1),
                    n_classes = class_num,
                    base_chns= config_net.get('base_feature_number', 16),
                    droprate=config_net.get('drop_rate', 0.2),
                    norm='in',
                    depth=config_net.get('depth', False),
                    dilation=config_net.get('dilation', 1),
                    separate_direction='axial'
                    )
    net = torch.nn.DataParallel(net, device_ids=[0, 1]).cuda()
    if config_train['load_weight']:
        weight = torch.load(config_train['model_path'], map_location=lambda storage, loc: storage)
        net.load_state_dict(weight)

    show_param(net)

    dice_eval = TestDiceLoss(n_class=class_num)
    loss_func = AttentionExpDiceLoss(n_class=class_num, alpha=0.5)
    show_loss = loss_visualize(class_num)

    Adamoptimizer = optim.Adam(net.parameters(), lr=lr, weight_decay= config_train.get('decay', 1e-7))
    Adamscheduler = torch.optim.lr_scheduler.StepLR(Adamoptimizer, step_size=10, gamma=0.9)

    # 4, start to train
    print('4.Start to train')
    dice_file = config_train['model_save_prefix'] + "_dice.txt"
    start_it  = config_train.get('start_iteration', 0)
    dice_save= np.zeros([config_train['maximal_epoch'], 2+class_num])  
    for n in range(start_it, config_train['maximal_epoch']):    
        train_loss_list, train_dice_list = np.zeros(config_train['train_step']//config_train['print_step']), np.zeros([config_train['train_step']//config_train['print_step'], class_num])
        valid_loss_list, valid_dice_list = np.zeros(valid_patient_number), np.zeros([valid_patient_number, class_num])


        optimizer = Adamoptimizer

        net.train()
        print('###train###\n')
        for step in range(config_train['train_step']):
            train_pair = trainData.get_subimage_batch()
            tempx = torch.FloatTensor(train_pair['images']).cuda()
            tempy = torch.FloatTensor(train_pair['labels']).cuda()
            # soft_tempy = get_soft_label(tempy.unsqueeze(1), class_num)
            predic = net(tempx)
            train_loss = loss_func(predic, tempy)
            optimizer.zero_grad()  
            train_loss.backward() 
            # torch.nn.utils.clip_grad_norm(net.parameters(), 10)
            optimizer.step()  
            if step%config_train['print_step']==0:
                train_loss = train_loss.cpu().data.numpy()
                train_loss_list[step//config_train['print_step']] = train_loss
                train_dice = dice_eval(predic, tempy)
                train_dice = train_dice.cpu().data.numpy()
                train_dice_list[step//config_train['print_step']] = train_dice
                print('train loss:', train_loss, ' train dice:', train_dice)
        Adamscheduler.step()

        print('###test###\n')
        with torch.no_grad():
            net.eval()
            for patient_order in range(valid_patient_number):
                valid_pair, patient_path = validData.get_list_img(patient_order)  
                clip_number = len(valid_pair['images'])    
                clip_height = config_data['test_data_shape'][0]  
                total_labels = valid_pair['labels'].cuda()
                predic_size = torch.Size([1, class_num])+total_labels.size()[1::]
                totalpredic = torch.zeros(predic_size).cuda()  

                for i in range(clip_number):
                    tempx = valid_pair['images'][i].cuda()
                    pred = net(tempx)
                    # pred[:, 0][tempx[:, 0] <= 0.0001] = 1   
                    if i < clip_number-1:
                        totalpredic[:, :, i * clip_height:(i + 1) * clip_height] = pred
                    else:
                        totalpredic[:, :, -clip_height::] = pred

                valid_dice = dice_eval(totalpredic, total_labels, show=True).cpu().data.numpy()
                valid_dice_list[patient_order] = valid_dice
                print(' valid dice:', valid_dice)


        batch_dice = [valid_dice_list.mean(axis=0), train_dice_list.mean(axis=0)]
        t = time.strftime('%X %x %Z')
        print(t, 'n', n, '\ndice:\n', batch_dice)
        show_loss.plot_loss(n, batch_dice)
        train_dice_mean = np.asarray([batch_dice[1][1::].mean(axis=0)])
        valid_dice_classes = batch_dice[0][1::]
        valid_dice_mean = np.asarray([valid_dice_classes.mean(axis=0)])
        batch_dice = np.append(np.append(train_dice_mean,
                               valid_dice_mean), valid_dice_classes)
        dice_save[n] = np.append(n, batch_dice)

        if batch_dice[1] > best_dice:
            best_dice = batch_dice[1]
            torch.save(net.state_dict(), config_train['model_save_prefix'] + "_{0:}.pkl".format(batch_dice[1]))
    
if __name__ == '__main__':
    config_file = str('config/train.txt')
    assert(os.path.isfile(config_file))
    train(config_file)
