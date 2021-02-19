import os
import sys

sys.path.append(".")
sys.path.append("..")

import torch
from torch import nn
from torch.nn import functional as F

import matplotlib.pyplot as plt

from config.config import cfg


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,  # 入力のチャネル数
            out_channels=6,  # 出力のチャネル数
            kernel_size=5,  # カーネルサイズ
            stride=1,  # ストライド
            padding=0,  # パディング
        )

        self.conv2 = nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2,)  # カーネルサイズ  # ストライド

        self.fc1 = nn.Linear(
            in_features=16 * 5 * 5, out_features=120,  # 全結合層への入力数  # 出力数
        )

        self.fc2 = nn.Linear(in_features=120, out_features=84)

        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.viwe(-1, 16 * 5 * 5)  # 1次元データに変換し、全結合層へ
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    # 畳み込み層の可視化用関数
    def plot_conv1(self, prefix_num=0):
        weights1 = self.conv1.weight
        weights1 = weights1.reshape(3 * 6, 5, 5)

        for i, weight in enumerate(weights1):
            plt.subplot(3, 6, i + 1)
            plt.imshow(weight.data.numpy(), cmap="winter")
            plt.tick_params(
                labelbottom=False,
                labelleft=False,
                labelright=False,
                labeltop=False,
                bottom=False,
                left=False,
                right=False,
                top=False,
            )

            plt.savefig(cfg.IMAGE_DIR + os.sep + "{}_conv1.png".format(prefix_num))
            plt.close()


    def train(self):
        epoch = 50



if __name__ == "__main__":

    net: LeNet = LeNet()
    criterion = nn.CrossEntropyLoss()  # 損失を計算
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)

    # 学習フィルタの可視化
    net.plot_conv1()