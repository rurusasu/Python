import os
import sys

sys.path.append(".")
sys.path.append("..")

import torch
from torch import nn, optim

from config.config import cfg
from read.cifar10 import load_cifar10
from model.LeNet import LeNet

def train():
    epoch = 50

    loader = load_cifar10()
    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    net: MyCNN = LeNet()
    criterion = nn.CrossEntropyLoss() # ロスの計算
    optimizer = optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

    #学習前のフィルタの可視化
    net.plot_conv1()


if __name__ == '__main__':
    train()