import torch
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision.datasets.folder import default_loader
from pathlib import Path
import os
import h5py
import sys
import numpy as np
from torchvision import transforms
from PIL import Image
import math
import cv2 as cv
from configs_argument import get_config
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def cv_imread(filepath):
    cv_img = cv.imdecode(np.fromfile(filepath,dtype=np.uint8),-1)
    if len(cv_img.shape)==3:
        cv_img = cv.cvtColor(cv_img, cv.COLOR_RGB2GRAY)
    return cv_img


class data_loader(Dataset):
    def __init__(self,mode="train"):
        self.train_path = r'D:\wy\滤泡-非滤泡\data\\' + mode +'.txt'  #r'D:\wy\滤泡-非滤泡\data\train.txt'
        self.mode = mode
        self.label = []  # self 全局变量
        self.image = []
        with open(self.train_path, 'rt') as f:  # 打开滤泡txt
            self.lines = f.readlines()
        for i in range(len(self.lines)):
            data_path = self.lines[i]
            a = data_path.split('|')
            image_path = a[0]
            mask_path = a[1]
            self.label.append(int(a[3][0]))
            image = cv_imread(image_path)
            mask = cv_imread(mask_path)
            image = torch.tensor(image)
            mask = torch.tensor(mask)
            imglist = [image, mask, image]  # 三个通道
            s = torch.stack(imglist)  # 对序列数据内部的张量进行扩维拼接
            self.image.append(s)   # img
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.2079,0.1085,0.2079],[0.2081,0.3096,0.2081])
        ])
        self.valid_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.2079,0.1085,0.2079],[0.2081,0.3096,0.2081])
        ])

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):

        label = self.label[index]
        image = self.image[index]
        if self.mode == "train":
            image = self.train_transform(image)
        else:
            image = self.valid_transform(image)


        return image, label


class ComboIter(object):
    """An iterator."""
    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.next() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    def __len__(self):
        return len(self.my_loader)


# 用  将两个loader结合起来
class ComboLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
    loaders: a list or tuple of pytorch DataLoader objects
    """
    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return ComboIter(self)

    def __len__(self):
        return min([len(loader) for loader in self.loaders])

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


def get_sampling_probabilities(class_count, mode='instance', ep=None, n_eps=None):
    '''
    Note that for progressive sampling I use n_eps-1, which I find more intuitive.
    If you are training for 10 epochs, you pass n_eps=10 to this function. Then, inside
    the training loop you would have sth like 'for ep in range(n_eps)', so ep=0,...,9,
    and all fits together.
    '''
    if mode == 'instance':
        q = 0
    elif mode == 'class':
        q = 1
    elif mode == 'sqrt':
        q = 0.5 # 1/2
    elif mode == 'cbrt':
        q = 0.125 # 1/8
    elif mode == 'prog':
        assert ep != None and n_eps != None, 'progressive sampling requires to pass values for ep and n_eps'
        relative_freq_imbal = class_count ** 0 / (class_count ** 0).sum()
        relative_freq_bal = class_count ** 1 / (class_count ** 1).sum()
        sampling_probabilities_imbal = relative_freq_imbal ** (-1)
        sampling_probabilities_bal = relative_freq_bal ** (-1)
        return (1 - ep / (n_eps - 1)) * sampling_probabilities_imbal + (ep / (n_eps - 1)) * sampling_probabilities_bal
    else: sys.exit('not a valid mode')

    relative_freq = class_count ** q / (class_count ** q).sum()  # class_count--这类的数据量  每类采样概率  和为1
    sampling_probabilities = relative_freq ** (-1)

    return sampling_probabilities



def modify_loader(class_count,loader, label,mode, ep=None, n_eps=None):
    class_count = np.array(class_count)
    target = torch.tensor(label)
    sampling_probs = get_sampling_probabilities(class_count, mode=mode, ep=ep, n_eps=n_eps)  # 计算每类的采样概率
    sampling_probs = torch.tensor(sampling_probs)
    ids = target.view(-1, 1)  # 将target reshape成1列，-1行不确定
    sample_weights = sampling_probs[ids.data.view(-1)]  # 每个样本采样的权重
    sample_weights = sample_weights.tolist()
    mod_sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights))  # 定义取batch方法
    mod_loader = DataLoader(loader.dataset, batch_size = loader.batch_size, sampler=mod_sampler, num_workers=loader.num_workers)
    return mod_loader


# 用
def get_combo_loader(class_count,loader, label,base_sampling='instance'):
    if base_sampling == 'instance':
        imbalanced_loader = loader
    else:
        imbalanced_loader = modify_loader(class_count,loader,label, mode=base_sampling)  # 不均衡

    balanced_loader = modify_loader(class_count,loader, label,mode='class')  # 均衡采样方法 class
    combo_loader = ComboLoader([imbalanced_loader, balanced_loader])  # 将均衡loader和不均衡loader结合起来
    return combo_loader



if __name__ == "__main__":

    dataset = data_loader("val")
    img_tensor = dataset[3]
    # print(img_tensor)

