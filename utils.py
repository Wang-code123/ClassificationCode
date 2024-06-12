import torch
from configs import get_config
import itertools
from tqdm import tqdm
from warmup_scheduler.scheduler import GradualWarmupScheduler
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import multilabel_confusion_matrix


# 多分类focal loss---加alpha
class MultiFocalLoss(torch.nn.Module):
    def __init__(self, class_num, Alpha=None, gamma=2):
        super(MultiFocalLoss, self).__init__()  # 子类继承父类方法
        if Alpha is None:
            self.Alpha = Variable(torch.ones(class_num,1))
        else:
            self.Alpha = Alpha  # 调节因子--控制类别不平衡
        self.gamma = gamma
        self.class_num = class_num

    def forward(self, input, labels):
        preds = F.softmax(input,dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        class_mask = F.one_hot(labels,self.class_num)  # 获取label的one-hot 编码
        ids= labels.view(-1,1)  # 将label reshape成1列， -1行不确定
        Alpha = torch.tensor(self.Alpha).cuda()  # 将alpha转换成tensor类型，cuda--GPU tensor
        Alpha = Alpha[ids.data.view(-1)]  # 将alpha中对应每个类的权重赋值给每个样本--根据label
        epsilon = 1e-7  # 防止为0
        pt = (preds*class_mask).sum(1).view(-1,1)+epsilon
        log_p = pt.log()
        loss = -Alpha*(torch.pow((1-pt),self.gamma).squeeze())*log_p.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        loss = loss.mean()
        return loss


# 多分类class-balanced focal loss
class CBMultiFocalLoss(torch.nn.Module):
    def __init__(self, class_num,num_per_cls, gamma=2, sigma=0.99):
        super(CBMultiFocalLoss, self).__init__()  # 子类继承父类方法
        self.gamma = gamma
        self.class_num = class_num
        self.sigma = sigma
        self.num_per_cls = num_per_cls

    def forward(self, input, labels):
        preds = F.softmax(input,dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        class_mask = F.one_hot(labels,self.class_num)  # 获取label的one-hot 编码
        ids = labels.view(-1,1)  # 将label reshape成1列， -1行不确定
        effective_num = 1.0 - np.power(self.sigma,self.num_per_cls)
        weights = (1.0-self.sigma)/np.array(effective_num)
        weights = weights/np.sum(weights)*int(self.class_num)
        weights = torch.tensor(weights).cuda()
        weights = weights[ids.data.view(-1)]
        epsilon = 1e-7  # 防止为0
        pt = (preds*class_mask).sum(1).view(-1,1)+epsilon
        log_p = pt.log()
        loss = -weights*(torch.pow((1-pt),self.gamma).squeeze())*log_p.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        loss = loss.sum()
        return loss


# focal tversky loss--多分类
class FocalTverskyLoss(torch.nn.Module):
    # sum_c（1-TI_c）^1/gamma   c--类别数  TI = (tp+1)/(tp+theta*fn+(1-theta)*fp+1)
    def __init__(self, theta=0.7, tau=0.75):
        super(FocalTverskyLoss, self).__init__()
        self.theta = theta
        self.tau = tau

    def forward(self, predict, target):
        y_pred = torch.max(predict, dim=1)[1]  # 预测的label
        y_pred11 = y_pred.cpu().numpy()  # 将tensor变成numpy
        target11 = target.cpu().numpy()
        mc_matrix = multilabel_confusion_matrix(y_pred11,target11)  # 计算多分类混淆矩阵 2*2*n_class  要求输入是numpy
        # 判断是否是三类，如果缺了，在对应类位置补全混淆矩阵[0,0][0,0]
        dd={0:False, 1:False, 2:False}
        for i in range(y_pred.shape[0]):
            dd[y_pred[i].item()]=True
            dd[target[i].item()] = True
        if(False in dd.values()):
            falist = [list(dd.keys())[list(dd.values()).index(False)]]
            for i in range(len(falist)):
                mc_matrix = np.insert(mc_matrix, falist[i], [[0,0],[0,0]], axis=0)
        fp = mc_matrix[:,0,1]
        fn = mc_matrix[:,1,0]
        tp = mc_matrix[:,1,1]
        tversky1 = (tp[0]+1) / (tp[0]+(1-self.theta)*fn[0]+self.theta*fp[0]+1)  # 0---tp/(tp+theta*fn+(1-theta)*fp)
        tversky2 = (tp[1]+1) / (tp[1] + self.theta * fn[1] + (1-self.theta) * fp[1]+1)  # 1
        tversky3 = (tp[2]+1) / (tp[2] + self.theta * fn[2] + (1-self.theta) * fp[2]+1)   # 2
        tversky1,tversky2,tversky3 = torch.tensor(tversky1,requires_grad=True).cuda(),\
                                     torch.tensor(tversky2,requires_grad=True).cuda(),\
                                     torch.tensor(tversky3,requires_grad=True).cuda()  # 转换成tensor
        loss = torch.pow((1 - tversky1), self.tau) + torch.pow((1 - tversky2), self.tau) +\
               torch.pow((1 - tversky3), self.tau)   # loss
        return loss


# CB-focal tversky loss--多分类
class CBFocalTverskyLoss(torch.nn.Module):
    # sum_c（1-TI_c）^1/gamma   c--类别数  TI = (tp+1)/(tp+theta*fn+(1-theta)*fp+1)
    def __init__(self, class_num, num_per_cls, theta=0.7, tau=0.75, sigma=0.99):
        super(CBFocalTverskyLoss, self).__init__()
        self.theta = theta
        self.tau = tau
        self.sigma = sigma
        self.class_num = class_num
        self.num_per_cls = num_per_cls

    def forward(self, predict, target):
        y_pred = torch.max(predict, dim=1)[1]  # 预测的label
        y_pred11 = y_pred.cpu().numpy()  # 将tensor变成numpy
        target11 = target.cpu().numpy()
        mc_matrix = multilabel_confusion_matrix(y_pred11,target11)  # 计算多分类混淆矩阵 2*2*n_class
        # 判断是否是三类，如果缺了，在对应类位置补全混淆矩阵[0,0][0,0]
        dd={0:False, 1:False, 2:False}
        for i in range(y_pred.shape[0]):
            dd[y_pred[i].item()]=True
            dd[target[i].item()] = True
        if(False in dd.values()):
            falist = [list(dd.keys())[list(dd.values()).index(False)]]
            for i in range(len(falist)):
                mc_matrix = np.insert(mc_matrix, falist[i], [[0,0],[0,0]], axis=0)
        fp = mc_matrix[:,0,1]
        fn = mc_matrix[:,1,0]
        tp = mc_matrix[:,1,1]
        tversky1 = (tp[0]+1) / (tp[0]+(1-self.theta)*fn[0]+self.theta*fp[0]+1)  # 0---tp/(tp+theta*fn+(1-theta)*fp)
        tversky2 = (tp[1]+1) / (tp[1] + self.theta * fn[1] + (1-self.theta) * fp[1]+1)  # 1
        tversky3 = (tp[2]+1) / (tp[2] + self.theta * fn[2] + (1-self.theta) * fp[2]+1)   # 2
        tversky1,tversky2,tversky3 = torch.tensor(tversky1,requires_grad=True).cuda(),\
                                     torch.tensor(tversky2,requires_grad=True).cuda(),\
                                     torch.tensor(tversky3,requires_grad=True).cuda()  # 转换成tensor
        effective_num = 1.0-np.power(self.sigma, self.num_per_cls)  # 有效数 --分母 1-sigma^n_y
        weights = (1.0-self.sigma)/np.array(effective_num)  # Class-Balanced loss  相当于之前的权重因子alpha
        weights = weights/np.sum(weights)*int(self.class_num)
        # 将权重归一化到[0,1]---防止 1-weights[0]*tversky1 小于0
        weights_norm = np.linalg.norm(weights)
        weights_new = []
        for i in range(len(weights)):
            weights_normlization = weights[i] / weights_norm
            weights_new.append(weights_normlization)
        weights = torch.tensor(weights_new).cuda()  # 将weights变成GPU的tensor
        loss = torch.pow((1 - weights[0]*tversky1), self.tau) + torch.pow((1 - weights[1]*tversky2), self.tau) +\
               torch.pow((1 - weights[2]*tversky3), self.tau)   # loss
        return loss


# CB-LDAMLoss
class LDAMLoss(torch.nn.Module):
    def __init__(self, num_per_cls, max_m=0.5, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(num_per_cls))  # 1/（n^1/4）  n--每类的样本数
        m_list = m_list * (max_m / np.max(m_list))  # C/（n^1/4）   常数C--max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor--CPU float--四位小数    torch.cuda.FloatTensor--GPU
        self.m_list = m_list
        assert s > 0
        self.s = s  # 缩放因子
        self.num_per_cls = num_per_cls

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)  # 和x维度一致全0的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # 将label one-hot编码

        index_float = index.type(torch.cuda.FloatTensor)  # torch.cuda.FloatTensor  转成float tensor
        a=index_float.transpose(0, 1)
        b=self.m_list[None, :]
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 点乘
        batch_m = batch_m.view((-1, 1))  # 变成1列
        x_m = x - batch_m  # Z_y- C/（n^1/4）

        output = torch.where(index, x_m, x)  # 合并两个tensor--在one-hot 1对应的位置换成x_m对应位置的元素，0对应的位置换成x对应的元素
        # 权重--Class-Balanced 系数
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, self.num_per_cls)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_per_cls)
        per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
        return F.cross_entropy(self.s * output, target, weight=per_cls_weights)


# 多分类CB-Focal Tversky+CB-LDAM loss
class CBFTLDAMLoss(torch.nn.Module):
    def __init__(self, class_num,num_per_cls, max_m=0.5, s=30, theta=0.7, tau=0.75, sigma=0.99):
        super(CBFTLDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(num_per_cls))  # 1/（n^1/4）  n--每类的样本数
        m_list = m_list * (max_m / np.max(m_list))  # C/（n^1/4）   常数C--max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor--CPU float--四位小数    torch.cuda.FloatTensor--GPU
        self.m_list = m_list
        assert s > 0
        self.s = s  # 缩放因子
        self.num_per_cls = num_per_cls
        self.theta = theta
        self.tau = tau
        self.sigma = sigma
        self.class_num = class_num


    def forward(self, predict, target):
        index = torch.zeros_like(predict, dtype=torch.uint8)  # 和x维度一致全0的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # 将label one-hot编码
        index_float = index.type(torch.cuda.FloatTensor)  # torch.cuda.FloatTensor  转成float tensor
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 点乘
        batch_m = batch_m.view((-1, 1))  # 变成1列
        x_m = predict - batch_m  # Z_y- C/（n^1/4）
        output = torch.where(index, x_m, predict)  # 合并两个tensor--在one-hot 1对应的位置换成x_m对应位置的元素，0对应的位置换成x对应的元素
        # 权重--Class-Balanced 系数
        effective_num = 1.0 - np.power(self.sigma, self.num_per_cls)  # 有效数 --分母 1-sigma^n_y
        weights = (1.0 - self.sigma) / np.array(effective_num)  # Class-Balanced loss  相当于之前的权重因子alpha
        weights = weights / np.sum(weights) * int(self.class_num)
        # 将权重归一化到[0,1]---防止 1-weights[0]*tversky1 小于0
        weights_norm = np.linalg.norm(weights)
        weights_new = []
        for i in range(len(weights)):
            weights_normlization = weights[i] / weights_norm
            weights_new.append(weights_normlization)
        weights = torch.cuda.FloatTensor(weights_new)  # 将weights变成CPU的tensor
        loss_ladm = F.cross_entropy(self.s * output, target, weight=weights)

        y_pred = torch.max(predict, dim=1)[1]  # 预测的label
        y_pred11 = y_pred.cpu().numpy()  # 将tensor变成numpy
        target11 = target.cpu().numpy()
        mc_matrix = multilabel_confusion_matrix(y_pred11,target11)  # 计算多分类混淆矩阵 2*2*n_class
        # 判断是否是三类，如果缺了，在对应类位置补全混淆矩阵[0,0][0,0]
        dd={0:False, 1:False, 2:False}
        for i in range(y_pred.shape[0]):
            dd[y_pred[i].item()]=True
            dd[target[i].item()] = True
        if(False in dd.values()):
            falist = [list(dd.keys())[list(dd.values()).index(False)]]
            for i in range(len(falist)):
                mc_matrix = np.insert(mc_matrix, falist[i], [[0,0],[0,0]], axis=0)
        fp = mc_matrix[:,0,1]
        fn = mc_matrix[:,1,0]
        tp = mc_matrix[:,1,1]
        tversky1 = (tp[0]+1) / (tp[0]+(1-self.theta)*fn[0]+self.theta*fp[0]+1)  # 0---tp/(tp+theta*fn+(1-theta)*fp)
        tversky2 = (tp[1]+1) / (tp[1] + self.theta * fn[1] + (1-self.theta) * fp[1]+1)  # 1
        tversky3 = (tp[2]+1) / (tp[2] + self.theta * fn[2] + (1-self.theta) * fp[2]+1)   # 2
        tversky1,tversky2,tversky3 = torch.tensor(tversky1,requires_grad=True),\
                                     torch.tensor(tversky2,requires_grad=True),\
                                     torch.tensor(tversky3,requires_grad=True)  # 转换成tensor
        loss_FT = torch.pow((1 - weights[0]*tversky1), self.tau) + torch.pow((1 - weights[1]*tversky2), self.tau) +\
               torch.pow((1 - weights[2]*tversky3), self.tau)   # loss
        loss = loss_ladm + loss_FT
        return loss


class FLDAMLoss(torch.nn.Module):
    def __init__(self, num_per_cls, class_num, gamma=2, max_m=0.5, s=30):
        super(FLDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(num_per_cls))  # 1/（n^1/4）  n--每类的样本数
        m_list = m_list * (max_m / np.max(m_list))  # C/（n^1/4）   常数C--max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor--CPU float--四位小数    torch.cuda.FloatTensor--GPU
        self.m_list = m_list
        assert s > 0
        self.s = s  # 缩放因子
        self.num_per_cls = num_per_cls
        self.class_num = class_num
        self.gamma = gamma

    def forward(self, input, target):
        index = torch.zeros_like(input, dtype=torch.uint8)  # 和x维度一致全0的tensor
        index.scatter_(1, target.data.view(-1, 1), 1)  # 将label one-hot编码

        index_float = index.type(torch.cuda.FloatTensor)  # torch.cuda.FloatTensor  转成float tensor
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 点乘
        batch_m = batch_m.view((-1, 1))  # 变成1列
        x_m = input - batch_m  # Z_y- C/（n^1/4）

        output = torch.where(index, x_m, input)  # 合并两个tensor--在one-hot 1对应的位置换成x_m对应位置的元素，0对应的位置换成x对应的元素
        preds = F.softmax(output, dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        class_mask = F.one_hot(target, self.class_num)  # 获取label的one-hot 编码
        ids = target.view(-1, 1)  # 将label reshape成1列， -1行不确定
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, self.num_per_cls)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_per_cls)
        per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
        per_cls_weights = per_cls_weights[ids.data.view(-1)]
        epsilon = 1e-7  # 防止为0
        pt = (preds*class_mask).sum(1).view(-1,1)+epsilon
        log_p = pt.log()
        loss = -per_cls_weights * (
            torch.pow((1 - pt), self.gamma).squeeze()) * log_p.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        # 权重--Class-Balanced 系数
        loss = loss.mean()
        return loss


class LMF(torch.nn.Module):
    def __init__(self, class_num, num_per_cls, mu=0.5, gamma=2, max_m=0.5, s=30):
        super(LMF, self).__init__()  # 子类继承父类方法
        self.gamma = gamma
        self.class_num = class_num
        self.num_per_cls = num_per_cls
        m_list = 1.0 / np.sqrt(np.sqrt(num_per_cls))  # 1/（n^1/4）  n--每类的样本数
        m_list = m_list * (max_m / np.max(m_list))  # C/（n^1/4）   常数C--max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor--CPU float--四位小数    torch.cuda.FloatTensor--GPU
        self.m_list = m_list
        assert s > 0
        self.s = s  # 缩放因子
        self.mu = mu

    def forward(self, input, labels):
        preds = F.softmax(input,dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        class_mask = F.one_hot(labels,self.class_num)  # 获取label的one-hot 编码
        epsilon = 1e-7  # 防止为0
        pt = (preds*class_mask).sum(1).view(-1,1)+epsilon
        log_p = pt.log()
        loss_FL = -(torch.pow((1-pt),self.gamma).squeeze())*log_p.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        loss_FL = loss_FL.mean()

        index = torch.zeros_like(input, dtype=torch.uint8)  # 和x维度一致全0的tensor
        index.scatter_(1, labels.data.view(-1, 1), 1)  # 将label one-hot编码
        index_float = index.type(torch.cuda.FloatTensor)  # torch.cuda.FloatTensor  转成float tensor
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 点乘
        batch_m = batch_m.view((-1, 1))  # 变成1列
        x_m = input - batch_m  # Z_y- C/（n^1/4）
        output = torch.where(index, x_m, input)  # 合并两个tensor--在one-hot 1对应的位置换成x_m对应位置的元素，0对应的位置换成x对应的元素
        # 权重--Class-Balanced 系数
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, self.num_per_cls)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_per_cls)
        per_cls_weights = torch.cuda.FloatTensor(per_cls_weights)
        loss_LDAM = F.cross_entropy(self.s * output, labels, weight=per_cls_weights)
        loss_LDAM = loss_LDAM.mean()
        loss = self.mu * loss_FL + (1 - self.mu) * loss_LDAM
        return loss


class CBFL_FLDAM(torch.nn.Module):
    def __init__(self, class_num, num_per_cls, mu=0.6, gamma=2, max_m=0.5, s=30, sigma=0.999):
        super(CBFL_FLDAM, self).__init__()  # 子类继承父类方法
        self.gamma = gamma
        self.class_num = class_num
        self.sigma = sigma
        self.num_per_cls = num_per_cls
        m_list = 1.0 / np.sqrt(np.sqrt(num_per_cls))  # 1/（n^1/4）  n--每类的样本数
        m_list = m_list * (max_m / np.max(m_list))  # C/（n^1/4）   常数C--max_m / np.max(m_list)
        m_list = torch.cuda.FloatTensor(m_list)  # 转成 tensor--CPU float--四位小数    torch.cuda.FloatTensor--GPU
        self.m_list = m_list
        assert s > 0
        self.s = s  # 缩放因子
        self.mu = mu

    def forward(self, input, labels):
        preds = F.softmax(input,dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        class_mask = F.one_hot(labels,self.class_num)  # 获取label的one-hot 编码
        ids = labels.view(-1,1)  # 将label reshape成1列， -1行不确定
        effective_num = 1.0 - np.power(self.sigma,self.num_per_cls)
        weights = (1.0-self.sigma)/np.array(effective_num)
        weights = weights/np.sum(weights)*int(self.class_num)
        weights = torch.tensor(weights).cuda()
        weights = weights[ids.data.view(-1)]
        epsilon = 1e-7  # 防止为0
        pt = (preds*class_mask).sum(1).view(-1,1)+epsilon
        log_p = pt.log()
        loss_CBFL = -weights*(torch.pow((1-pt),self.gamma).squeeze())*log_p.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        loss_CBFL = loss_CBFL.mean()

        index = torch.zeros_like(input, dtype=torch.uint8)  # 和x维度一致全0的tensor
        index.scatter_(1, labels.data.view(-1, 1), 1)  # 将label one-hot编码
        index_float = index.type(torch.cuda.FloatTensor)  # torch.cuda.FloatTensor  转成float tensor
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))  # 点乘
        batch_m = batch_m.view((-1, 1))  # 变成1列
        x_m = input - batch_m  # Z_y- C/（n^1/4）
        output = torch.where(index, x_m, input)  # 合并两个tensor--在one-hot 1对应的位置换成x_m对应位置的元素，0对应的位置换成x对应的元素
        preds1 = F.softmax(output, dim=1)  # 预测概率--tensor   按行  每个样本关于每类的概率
        pt1 = (preds1 * class_mask).sum(1).view(-1, 1) + epsilon
        log_p1 = pt.log()
        loss_LDAM = -weights * (torch.pow((1 - pt1), self.gamma).squeeze()) * log_p1.squeeze()  # .squeeze()--减少一个维度，和alpha 维度相同
        # 权重--Class-Balanced 系数
        loss_LDAM = loss_LDAM.mean()
        loss = self.mu*loss_CBFL + (1-self.mu)*loss_LDAM
        return loss


class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


# MixUp 数据增强--生成混合图像
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    # class_mask = torch.nn.functional.one_hot(y, class_num)  # one-hot 编码
    mixed_x = lam * x + (1 - lam) * x[index, :]  # 生成混合图像
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# CutMix 生成裁剪框 输入为：样本的size和生成的随机lambda值
def rand_bbox(size, lam):
    W = size[2]  # 样本的宽
    H = size[3]  # 样本的高
    cut_rat = np.sqrt(1. - lam)
    # box 的宽和高 rw rh
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform  box的中心点rx ry
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 限制坐标区域不超过样本大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 返回裁剪box的坐标值
    return bbx1, bby1, bbx2, bby2


# CutMix 数据增强--生成混合图像
def cutmix_data(x, y, alpha=1.0):
    '''Returns generated inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)  # 随机生成lambda
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()  # 找到两个随机样本

    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)  # 生成裁剪区域box
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]  # 将样本A中的box区域替换成样本B中的box区域
    # 根据裁剪区域坐标框的值调整lambda值
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    # class_mask = torch.nn.functional.one_hot(y, class_num)  # one-hot 编码
    y_a, y_b = y, y[index]  # 一个batch   batch中的某一张
    return x, y_a, y_b, lam


def get_mean_std(dataset:torch.utils.data.Dataset,batch_size,num_workers,samples:int=224):
    if samples is not None and len(dataset)>samples:
        indices = np.random.choice(len(dataset),samples,replace=False)
        dataset = torch.utils.data.Subset(dataset,indices)
    dataloader = torch.utils.data.DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=False)

    n = 0
    s1 = 0
    s2 = 0
    for(x, *_) in tqdm(dataloader):
        x = x.transpose(0,1).contiguous().view(3,-1)
        n += x.shape[1]
        s1 += torch.sum(x, dim=1).numpy()
        s2 += torch.sum(x**2, dim=1).numpy()
    mean = s1/n
    std = np.sqrt(s2/n-mean**2)
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)

    return mean, std
