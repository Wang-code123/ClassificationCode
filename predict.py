import os
import json
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from model import resnet34,resnet101,resnet50
from Data_loader_modify import data_loader
import sklearn.metrics as metrics
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.metrics import cohen_kappa_score as kappa
from sklearn.metrics import balanced_accuracy_score as bal_acc


gpu_name = '0'  # Gpu 名称
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_name


def main():
    test_dataset = data_loader(mode = "test")  # 导入验证数据
    test_num = len(test_dataset)
    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    # create model
    model = resnet50(num_classes=3)  # 分类数量   resnet34

    # load model weights
    weights_path = "./compareLoss/FinalWeight/CBFL_FLDAM_0.9_1.5_0.05_cutmix/80.pth"  # 在验证集上结果好的参数
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    model.load_state_dict(torch.load(weights_path,map_location='cuda:0'))
    model = model.cuda()


    model.eval()
    acc = 0.0
    with torch.no_grad():
        # predict class
        test_bar = tqdm(test_loader, file=sys.stdout)
        all_preds = []
        all_labels = []
        all_probs = []
        prob = []
        all_image_path=[]
        for test_data in test_bar:
            test_images, test_labels, test_image_path = test_data
            paaaath = [path.split("\\")[-1] for path in test_image_path]
            all_image_path.extend(paaaath)

            output = model(test_images.cuda())
            predict = torch.softmax(output, dim=1)  # 预测为每类的概率值
            predict_cla = torch.argmax(predict, dim=1)  # 最大概率值对应的label
            acc += torch.eq(predict_cla, test_labels.cuda()).sum().item()
            all_preds.extend(list(predict_cla.detach().cpu().numpy()))  # 预测的label
            all_labels.extend(list(test_labels.detach().cpu().numpy()))  # 真实label
            all_probs.extend(list(predict.detach().cpu().numpy()))  # 预测的概率 3*n矩阵
            prob.extend(list(predict[:,1].detach().cpu().numpy()))  # 预测为滤泡的概率

    m = metrics.confusion_matrix(np.array(all_labels), np.array(all_preds))
    print(m)
    cnf_matrix = metrics.multilabel_confusion_matrix(np.array(all_labels), np.array(all_preds))  # 多分类混淆矩阵
    print(cnf_matrix)
    tn = cnf_matrix[:, 0, 0]
    fp = cnf_matrix[:, 0, 1]
    fn = cnf_matrix[:, 1, 0]
    tp = cnf_matrix[:, 1, 1]
    sen = tp / (tp + fn)  # 敏感性
    tey = tn / (fp + tn)  # 特异性
    fpr = fp / (fp + tn)
    fnr = fn / (tp + fn)
    test_acc = acc / test_num  # 准确性
    f1 = f1_score(all_labels, all_preds, average='macro')  # 计算每类f1求平均值   未加权
    mcc = matthews_corrcoef(all_labels, all_preds)  # 马修斯相关系数 【-1，1】1完美预测
    k = kappa(all_labels, all_preds, weights='quadratic')  # kappa系数
    bacc = bal_acc(all_labels, all_preds)  # 平衡精度  处理不平衡数据集  每个类获得的平均召回率
    mean_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')  # 多分类 multi_class='ovr'
    print('bacc: %.3f  sensitivity: %.3f  %.3f  %.3f specificity: %.3f %.3f %.3f' %
          (bacc, sen[0], sen[1], sen[2], tey[0], tey[1], tey[2]))  # 多分类输出


if __name__ == '__main__':
    main()
