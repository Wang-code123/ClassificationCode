import argparse
from pathlib import Path
import pprint


# project_dir = Path(__file__).resolve().parent
# # dataset_dir = Path('/data1/jysung710/tmp_sum/360video/').resolve()
# # video_list = ['360airballoon', '360parade', '360rowing', '360scuba', '360wedding']
# # save_dir = Path('/data1/jmcho/SUM_GAN/')
# # score_dir = Path('/data1/common_datasets/tmp_sum/360video/results/SUM-GAN/')
# dataset_dir = Path('./SumMe/ImgSeq/').resolve()
# save_dir = Path('./SumMe/result/').resolve()
# score_dir = Path('./SumMe/result/').resolve()
# def str2bool(v):
#     """string to boolean"""
#     if v.lower() in ('yes', 'true', 't', 'y', '1'):
#         return True
#     elif v.lower() in ('no', 'false', 'f', 'n', '0'):
#         return False
#     else:
#         raise argparse.ArgumentTypeError('Boolean value expected.')
import shutil
from pathlib import Path
def mkdirIfNExi(path):
    if path.exists():  # 判断所在目录下是否有该文件名的文件夹
        shutil.rmtree(path)
        path.mkdir(parents=True)
    else:
        path.mkdir(parents=True)
class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        for k, v in kwargs.items():
            setattr(self, k, v)  # 设置属性k的值为v
        self.save_path = './'+self.loss_function+'_'+str(self.mu)  +'_'+str(self.gamma) +'_'+str(self.Beta)
        sspath = Path(self.save_path).resolve()
        mkdirIfNExi(sspath)


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--n_epochs', type=int, default=100)  # epochs
    parser.add_argument('--gpu_name', type=str, default='0')  # gpu
    parser.add_argument('--batch_size', type=int, default=32)  # 32
    parser.add_argument('--lr', type=float, default=1e-4)  # 学习率
    parser.add_argument('--optim', type=str, default='sgd')  # 优化器
    parser.add_argument('--lr_scheduler', type=str, default='cosine')  # 调整学习率的方法
    parser.add_argument('--loss_function', type=str, default='CrossEntropyLoss')  # 损失函数
    parser.add_argument('--Alpha', nargs = '+', type=float, default=[0.7,0.1,0.2])  # 多分类focal loss 调节因子
    parser.add_argument('--gamma', type=float, default=1.5)  # focal loss 调节因子
    parser.add_argument('--sigma', type=float, default=0.999)  # CBfocal loss 参数
    parser.add_argument('--theta', type=float, default=0.7)  # Focal tversky loss 参数--调节假阳和假阴
    parser.add_argument('--tau', type=float, default=0.75)  # Focal tversky loss 参数--调节样本不均衡
    parser.add_argument('--Beta', type=float, default=0.2)  # 数据增强 随机生成 lambda
    parser.add_argument('--mu', type=float, default=0.9)  # 调节Focal loss 和LDAM loss 权重
    parser.add_argument('--data_aug', type=str, default='CutMix')  # 数据增强的方法
    parser.add_argument('--class_num', type=int, default=3)  # 类别数
    parser.add_argument('--sampling', type=str, default='instance', help='sampling mode(instance,class,sqrt,prog)')  # 类别数

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()

