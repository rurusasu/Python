import os
import argparse
import datetime

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def makedir(src: str, dir_name: str) -> str:
    """dst で指定された位置に dir_name で指定された名前のディレクトリを作成する関数．既に同名のディレクトリが存在する場合は，現在の「年, 月, 日_時, 分, 秒」を作成したい付ディレクトリ名の後ろに連結する．

    Args:
        src(str): ディレクトリを作成する場所の親ディレクトリまでのパス
        dir_name(str): 作成するディレクトリ名
    Return:
        dst(str): 作成したディレクトリの絶対パス
    """
    dst = os.path.join(src, dir_name)
    if os.path.exists(dst): # 存在する場合
        now = datetime.datetime.now()
        dst = dst + '_' + now.strftime('%Y%m%d_%H%M%S')
    os.mkdir(dst)

    return dst

class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val *n
        self.count += n
        self.avg = self.sum / self.count