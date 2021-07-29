import os
import sys

sys.path.append(".")
sys.path.append("..")
import json

import torch
from torch import nn, optim, smooth_l1_loss
from torch.nn import DataParallel

from config.config import cfg
from utils.net_utils import compute_precision_recall, load_model

with open(os.path.join(cfg.CONFIG_DIR, "linemod_train.json"), "r") as f:
    train_cfg = json.load(f)
train_cfg["model_name"] = "{}_{}".format("cat", train_cfg["model_name"])


class NetWrapper(nn.Module):
    def __init__(seld, net):
        super(NetWrapper, self).__init__()
        self.net = net
        # self.criterion = nn.CrossEntropyLoss(reduce = False)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, img, mask, vertex, vertex_weights):
        seg_pred, vertex_pred = self.net(img)
        loss_seg = self.criterion(seg_pred, mask)
        loss_seg = torch.mean(loss_seg.view(loss_seg.shape[0], -1), 1)
        loss_vertex = smooth_l1_loss(vertex_pred, vertex, vertex_weights, reduce=False)
        precision, recall = compute_precision_recall(seg_pred, mask)
        return seg_pred, vertex_pred, loss_seg, loss_vertex, precision, recall


def demo():
    net = Resnet18_8s(ver_dim=vote_num * 2, seg_dim=2)
    net = NetWrapper(net).cuda()
    net = DataParallel(net)

    optimizer = optim.Adam(net.parameters(), lr=train_cfg["lr"])
    model_dir = os.path.join(cfg.MODEL_DIR, "cat_demo")
    load_model(net.module.net, optimizer, model_dir, -1)
    data, points_3d, bb8_3d = read_data()
