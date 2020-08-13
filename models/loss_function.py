# coding:utf8
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import eq, sum, gt   # eq返回相同元素索引,gt返回大于给定值索引
from torch.nn import init
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


class TestDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(TestDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, input, target, show=False):
        smooth = 0.00001
        batch_size = input.size(0)
        input = torch.max(input, 1)[1]
        input = self.one_hot_encoder(input).contiguous().view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)
        inter = torch.sum(torch.sum(input * target, 2), 0) + smooth
        union1 = torch.sum(torch.sum(input, 2), 0) + smooth
        union2 = torch.sum(torch.sum(target, 2), 0) + smooth


        '''
        为避免当前训练图像中未出现的器官影响dice,删除dice大于0.98的部分
        '''
        andU = 2.0 * inter / (union1 + union2)
        score = andU

        return score.float()

class SoftDiceLoss(nn.Module):
    def __init__(self, n_class):
        super(SoftDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, input, target):
        '''
        :param input: the prediction, batchsize*n_class*depth*length*width
        :param target: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 0.01
        batch_size = input.size(0)

        input = input.view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)

        inter = torch.sum(input * target, 2) + smooth
        union1 = torch.sum(input, 2) + smooth
        union2 = torch.sum(target, 2) + smooth

        andU = torch.sum(2.0 * inter/(union1 + union2))
        score = 1 - andU/(batch_size*self.n_class)

        return score

class FocalLoss(nn.Module):
    def __init__(self, n_class):
        super(FocalLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class

    def forward(self, input,target):
        '''
        :param input: the prediction, batchsize*n_class*depth*length*width
        :param target: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        batch_size = input.size(0)
        input = input.view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)
        volume = input.size(2)
        score = -torch.sum(target*(1-input)**2*torch.log10(input))/volume

        return score

class Focal_and_Dice_loss(nn.Module):
    def __init__(self, n_class, lamda):
        super(Focal_and_Dice_loss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.lamda = lamda
        self.FocalLoss = FocalLoss(n_class)
        self.SoftDiceloss = SoftDiceLoss(n_class)

    def forward(self, input, target):
        '''
        :param input: the prediction, batchsize*n_class*depth*length*width
        :param target: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        score = self.lamda*self.FocalLoss(input, target)+self.n_class*self.SoftDiceloss(input, target)
        return score


class AttentionDiceLoss(nn.Module):
    def __init__(self, n_class, alpha):
        super(AttentionDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.alpha = alpha
    def forward(self, input, target):
        '''
        :param input: the prediction, batchsize*n_class*depth*length*width
        :param target: the groundtruth, batchsize*depth*length*width
        :return: loss
        '''
        smooth = 0.01
        batch_size = input.size(0)

        input = input.view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)
        attentioninput = torch.exp((input - target) / self.alpha) * input
        inter = torch.sum(attentioninput * target, 2) + smooth
        union1 = torch.sum(attentioninput, 2) + smooth
        union2 = torch.sum(target, 2) + smooth

        andU = torch.sum(2.0 * inter / (union1 + union2))
        score = batch_size * self.n_class - andU

        return score


class ExpDiceLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001):
        super(ExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        smooth = 1
        self.Ldice = Ldice(n_class, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, input, target):
        '''
        :param input: batch*class*depth*length*height or batch*calss*length*height
        :param target: batch*depth*length*height or batch*length*height
        :return:
        '''
        smooth = 1
        batch_size = input.size(0)
        realinput = input
        realtarget = target

        input = input.view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)
        label_sum = torch.sum(target[:, 1::], 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(input, target, batch_size)   #
        Lcross = self.Lcross(realinput, realtarget, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp


class AttentionExpDiceLoss(nn.Module):
    def __init__(self, n_class, alpha, gama=0.0001):
        super(AttentionExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = [1,1]
        self.alpha = alpha
        smooth = 1
        self.Ldice = Ldice(n_class-1, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, input, target):
        '''
        :param input: batch*class*depth*length*height or batch*calss*length*height
        :param target: batch*depth*length*height or batch*length*height
        :param dis: batch*class*depth*length*height or batch*calss*length*height
        :return:
        '''
        smooth = 1
        batch_size = input.size(0)
        realinput = input
        realtarget = target
        input = input.view(batch_size, self.n_class, -1)[:, 1::]
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)[:, 1::]
        attentionseg = torch.exp((input - target)/self.alpha) * input
        label_sum = torch.sum(target, 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5  # 各label占总非背景类label比值的开方
        Ldice = self.Ldice(attentionseg, target, batch_size)   #
        Lcross = self.Lcross(realinput, realtarget, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp


class BatchExpDiceLoss(nn.Module):
    def __init__(self, n_class, weights=[1, 1], gama=0.0001):
        super(BatchExpDiceLoss, self).__init__()
        self.one_hot_encoder = One_Hot(n_class).forward
        self.n_class = n_class
        self.gama = gama
        self.weight = weights
        smooth = 1
        self.BatchLdice = BatchLdice(n_class, smooth)
        self.Lcross = Lcross(n_class)
    def forward(self, input, target):
        '''
        :param input: batch*class*depth*length*height or batch*calss*length*height
        :param target: batch*depth*length*height or batch*length*height
        :return:    batch*ExpDice
        '''
        smooth = 1
        batch_size = input.size(0)
        realinput = input
        realtarget = target

        input = input.view(batch_size, self.n_class, -1)
        target = self.one_hot_encoder(target).contiguous().view(batch_size, self.n_class, -1)
        label_sum = torch.sum(target[:, 1::], 2) + smooth  # 非背景类label各自和
        Wl = (torch.sum(label_sum) / torch.sum(label_sum, 0))**0.5
        Ldice = self.BatchLdice(input, target, batch_size)
        Lcross = self.Lcross(realinput, realtarget, Wl, label_sum)
        Lexp = self.weight[0] * Ldice + self.weight[1] * Lcross
        return Lexp


class One_Hot(nn.Module):
    def __init__(self, depth):
        super(One_Hot, self).__init__()
        self.depth = depth
        self.ones = torch.eye(depth).cuda()  # torch.sparse.torch.eye
                                             # eye生成depth尺度的单位矩阵

    def forward(self, X_in):
        '''
        :param X_in: batch*depth*length*height or batch*length*height
        :return: batch*class*depth*length*height or batch*calss*length*height
        '''
        n_dim = X_in.dim()  # 返回dimension数目
        output_size = X_in.size() + torch.Size([self.depth])   # 增加一个class通道
        num_element = X_in.numel()  # 返回element总数
        X_in = X_in.data.long().view(num_element)   # 将target拉伸为一行
        out1 = Variable(self.ones.index_select(0, X_in))
        out = out1.view(output_size)
        return out.permute(0, -1, *range(1, n_dim)).squeeze(dim=2).float()  # permute更改dimension顺序

    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]


class Ldice(nn.Module):
    def __init__(self, smooth, n_class):
        super(Ldice, self).__init__()
        self.smooth = smooth
        self.n_class = n_class

    def forward(self, input, target, batch_size):
        '''
        Ldice
        '''
        inter = torch.sum(input * target, 2) + self.smooth
        union1 = torch.sum(input, 2) + self.smooth
        union2 = torch.sum(target, 2) + self.smooth
        dice = 2.0 * inter / (union1 + union2)
        logdice = -torch.log(dice)
        expdice = torch.sum(logdice) # ** self.gama
        Ldice = expdice / (batch_size*self.n_class)
        return Ldice


class Lcross(nn.Module):
    def __init__(self, n_class):
        super(Lcross, self).__init__()
        self.n_class = n_class
    def forward(self, realinput, realtarget, Wl, label_sum):
        '''
        realinput:
        realtarget:
        Wl: 各label占总非背景类label比值的开方
        '''
        Lcross = 0
        for i in range(1, self.n_class):
            mask = realtarget == i
            if torch.sum(mask).item() > 0:
                ProLabel = realinput[:, i][mask.detach()]
                LogLabel = -torch.log(ProLabel)
                ExpLabel = torch.sum(LogLabel)  # **self.gama
                Lcross += Wl[i - 1] * ExpLabel
        Lcross = Lcross / torch.sum(label_sum)

        return Lcross

