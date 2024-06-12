import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
from Data_loader_modify import data_loader, get_combo_loader
from model import resnet50
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import balanced_accuracy_score as bal_acc
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter
from configs_argument import get_config
from utils import FocalLoss, GradualWarmupSchedulerV2, mixup_data, cutmix_data, MultiCEFocalLoss, \
    CBMultiFocalLoss, FocalTverskyLoss,LDAMLoss,CBFocalTverskyLoss,CBFTLDAMLoss,get_mean_std,CBFL_FLDAM,rand_bbox
from collections import Counter


gpu_name = '0'  # Gpu 名称
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name


def cross_entropy_loss(input: torch.Tensor,
                             target: torch.Tensor
                             ) -> torch.Tensor:
    return -(input.log_softmax(dim=-1) * target).sum(dim=-1).mean()


def main():
    config = get_config(mode='train', preprocessed=False)  # 实例化get_config这个函数
    write = SummaryWriter()
    train_dataset = data_loader(mode='train')  # 导入训练数据
    lable111 = train_dataset.label  # 训练样本的所有label
    num_per_cls = [lable111.count(0),lable111.count(1),lable111.count(2)]  # 计算每类的样本数
    train_num = len(train_dataset)

    batch_size = config.batch_size  # batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)
    validate_dataset = data_loader(mode = "val")  # 导入验证数据
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    
    net = resnet50()
    # load pretrain weights
    model_weight_path = "./resnet50-19c8e357.pth"  # 迁移学习 ImageNet权重参数
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # change fc layer structure 全连接层
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 3)
    net = net.cuda()

    # define loss function  损失函数
    if config.loss_function == 'FocalLoss':
        loss_function = FocalLoss(config.alpha, config.gamma)  # FocalLoss
    elif config.loss_function == 'CrossEntropyLoss':
        loss_function = nn.CrossEntropyLoss()  # 交叉熵损失函数
    elif config.loss_function == 'MultiCEFocalLoss':
        loss_function = MultiCEFocalLoss(class_num=config.class_num, gamma=config.gamma, Alpha=config.Alpha)  # 多分类FocalLoss
    elif config.loss_function == 'CBMultiFocalLoss':
        loss_function = CBMultiFocalLoss(class_num=config.class_num, num_per_cls=num_per_cls,
                                         gamma=config.gamma, sigma=config.sigma)  # 多分类Class-Balanced FocalLoss
    elif config.loss_function == 'FocalTverskyLoss':
        loss_function = FocalTverskyLoss(theta=config.theta, tau=config.tau)  # 多分类FocalTverskyLoss
    elif config.loss_function == 'CBFocalTverskyLoss':
        loss_function = CBFocalTverskyLoss(class_num=config.class_num, num_per_cls=num_per_cls,
                                           theta=config.theta, tau=config.tau, sigma=config.sigma)  # 多分类CB-FocalTverskyLoss
    elif config.loss_function == 'LDAMLoss':
        loss_function = LDAMLoss(num_per_cls=num_per_cls, max_m=0.5,s=30)  # 多分类LDAM loss
    elif config.loss_function == 'CBFTLDAMLoss':
        loss_function = CBFTLDAMLoss(class_num=config.class_num, num_per_cls=num_per_cls,max_m=0.5,s=30,
                                     theta=config.theta, tau=config.tau, sigma=config.sigma)  # 多分类CB-Focal Tversky+CB-LDAM loss
    elif config.loss_function == 'CBFL_FLDAM':
        loss_function = CBFL_FLDAM(class_num=config.class_num, num_per_cls=num_per_cls, mu=config.mu,
                                   gamma=config.gamma, max_m=0.5, s=30, sigma=0.999 )  # CBFocalLoss + CBFLDAM--多分类
    else:
        print('No loss function')

    # construct an optimizer 优化
    params = [p for p in net.parameters() if p.requires_grad]

    # 不同的优化方法
    if config.optim == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.lr, betas=(0.9, 0.99), weight_decay=0.0002)
    elif config.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=config.lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise NotImplementedError

    # 不同的学习率调整方法
    if config.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    elif config.lr_scheduler == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, config.n_epochs - 1)
    else:
        raise NotImplementedError
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=1,
                                                after_scheduler=lr_scheduler)

    # MixUp+CutMix 数据增强的loss  按lambda值分配权重
    def mixup_loss(loss_function, pred, y_a, y_b, lam):
        return lam * loss_function(pred, y_a) + (1 - lam) * loss_function(pred, y_b)

    # 开始训练
    epochs = config.n_epochs
    best_acc = 0.0
    train_steps = len(train_loader)
    sensi=[]
    spec = []
    accu = []
    file = open(config.save_path+"/result.txt", 'w')  # 保存每次的数据结果
    since = time.time()
    for epoch in range(epochs):
        # Modify sampling
        combo_loader = get_combo_loader(class_count=num_per_cls, loader=train_loader, label=lable111,
                                        base_sampling=config.sampling)  # 根据采样方法选择loader
        net.train()
        running_loss = 0.0
        train_bar = tqdm(combo_loader, file=sys.stdout)  # 进度条
        for step, data in enumerate(train_bar):
            images, labels = data[0][0],data[0][1]  # 图像和label
            balanced_images, balanced_labels = data[1][0],data[1][1]
            images, labels = images.cuda(), labels.squeeze().cuda()
            balanced_images, balanced_labels = balanced_images.cuda(), balanced_labels.squeeze().cuda()
            lam = np.random.beta(a=config.Beta, b=1)
            # CutMix 数据增强
            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)  # 生成裁剪区域box
            images[:, :, bbx1:bbx2, bby1:bby2] = balanced_images[:, :, bbx1:bbx2, bby1:bby2]  # 将样本A中的box区域替换成样本B中的box区域
            mixed_labels1 = (1 - lam) * labels + lam * balanced_labels
            del balanced_images
            del balanced_labels
            # 数据增强 Mixup/ cutmix
            if np.random.rand() > 1:  # 让原图像和生成的混合图像都参与训练，大于0.5做增强，小于是原图 ;  1--不做增强
                if config.data_aug == 'MixUp':
                    images, targets_a, targets_b, lam = mixup_data(images, labels,
                                                                   config.Beta)
                elif config.data_aug == 'CutMix':
                    images, targets_a, targets_b, lam = cutmix_data(images, labels,
                                                                   config.Beta)
                logits = net(images)  # 将生成的训练样本放进模型训练
                loss = mixup_loss(loss_function, logits, targets_a, targets_b, lam)  # 计算损失函数

            else:
                logits = net(images)
                loss = loss_function(logits, mixed_labels1.long())

            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            end = time.time()
            hours,rem = divmod(end - since,3600)  # 时间结束
            minutes, seconds = divmod(rem, 60)
            print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
                hours,minutes, seconds))

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f} ".format(epoch + 1,
                                                                     epochs,
                                                                     loss)

        # validate 验证
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            all_preds = []
            all_labels = []
            all_probs = []
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.cuda())
                probs = outputs.softmax(dim=1)  # 输出二维tensor，一个样本预测为每类的概率值，和为1
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.cuda()).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                all_preds.extend(list(predict_y.detach().cpu().numpy()))  # 预测的label
                all_labels.extend(list(val_labels.detach().cpu().numpy()))  # 真实label
                all_probs.extend(list(probs.detach().cpu().numpy()))
        cnf_matrix = metrics.multilabel_confusion_matrix(np.array(all_labels), np.array(all_preds))  # 多分类混淆矩阵
        print(cnf_matrix)
        tn = cnf_matrix[:, 0, 0]
        fp = cnf_matrix[:, 0, 1]
        fn = cnf_matrix[:, 1, 0]
        tp = cnf_matrix[:, 1, 1]

        sen = tp / (tp + fn)  # 每类的敏感性，n类有n个
        tey = tn / (fp + tn)  # 每类的特异性
        val_accurate = acc / val_num  # 准确性
        f1 = f1_score(all_labels, all_preds, average='macro')  # 计算每类f1求平均值   未加权
        mcc = matthews_corrcoef(all_labels, all_preds)  # 马修斯相关系数 【-1，1】1完美预测
        k = kappa(all_labels, all_preds, weights='quadratic')  # kappa系数
        bacc = bal_acc(all_labels, all_preds)  # 平衡精度  处理不平衡数据集  每个类获得的平均召回率
        mean_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  # 多分类 multi_class='ovr'
        val_accurate = acc / val_num  # 准确性
        print( '[epoch %d] train_loss: %.3f  val_accuracy: %.3f  sensitivity: %.3f  %.3f  %.3f specificity: %.3f %.3f %.3f' %
            (epoch + 1, running_loss / train_steps, val_accurate, sen[0], sen[1], sen[2], tey[0], tey[1], tey[2]))
        sensi.append(sen)
        spec.append(tey)
        accu.append(val_accurate)
        file.write(str(epoch) + "|" + str(format(sen[0], '.3f')) + " " + str(format(sen[1], '.3f')) + " " + str(
            format(sen[2], '.3f')) + "|"
                   + str(format(tey[0], '.3f')) + " " + str(format(tey[1], '.3f')) + " " + str(format(tey[2], '.3f'))
                   + "|" + str(format(f1, '.3f')) + "|" + str(format(mcc, '.3f')) + "|" + str(
            format(bacc, '.3f')) + "|" + str(format(mean_auc, '.3f')) + '\n')  # 多分类保存的内容
        print(cnf_matrix)
        save_path = config.save_path + '/' + str(epoch) + '.pth'  # 保存每个epoch的参数
        torch.save(net.state_dict(), save_path)
        scheduler_warmup.step()
        if epoch == 2:
            scheduler_warmup.step()  # bug workaround
        else:
            lr_scheduler.step()

    file.close()
    print('Finished Training')


if __name__ == '__main__':
    main()
