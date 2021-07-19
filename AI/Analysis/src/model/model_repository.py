import sys
from matplotlib.pyplot import imshow

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import torch
from torch import nn

from src.model.resnet import resnet18


class ResNet18_8s(nn.Module):
    def __init__(
        self,
        ver_dim,
        seg_dim,
        fcdim=256,
        s16dim=256,
        s8dim=128,
        s4dim=64,
        s2dim=32,
        raw_dim=32,
    ):
        super(ResNet18_8s, self).__init__()

        # プリトレーニングされたウェイトを読み込み、avgプール層を削除し、出力ストライドを8とする
        resnet18_8s = resnet18(
            pretrained=True,
            fully_conv=True,
            output_stride=8,
            remove_avg_pool_layer=True,
        )

        self._ver_dim = ver_dim
        self.seg_dim = seg_dim

        self.resnet18_8s = resnet18_8s

        # ResNetの FC層を置き換えるための層
        self.fc = nn.Sequential(
            nn.Conv2d(
                resnet18_8s.inplanes,
                fcdim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True),
        )

        # x16s -> 256
        self.conv16s = nn.Sequential(
            nn.Conv2d(256 + fcdim, s16dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s16dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up32sTo16s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x8s -> 128
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up16sTo8s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x4s -> 64
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up8sTo4s = nn.UpsamplingBilinear2d(scale_factor=2)

        # x2s -> 64
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True),
        )
        self.up4sTo2s = nn.UpsamplingBilinear2d(scale_factor=2)

        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1),
        )
        self.up2sToraw = nn.UpsamplingBilinear2d(scale_factor=2)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        # ResNet の FC 層を置き換える
        self.resnet18_8s.fc = self.fc

        # x -> [32, 3, 256, 256]
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)
        # x2s -> [32, 64, 128, 128]
        # x4s -> [32, 64, 64, 64]
        # x8s -> [32, 128, 32, 32]
        # x16s -> [32, 256, 16, 16]
        # x32s -> [32, 512, 8, 8]
        # xfc -> [32, 256, 8, 8]

        fm = self.up32sTo16s(xfc)  # fm -> [32, 256, 16, 16]
        fm = self.conv16s(torch.cat([fm, x16s], 1))

        fm = self.up16sTo8s(fm)
        fm = self.conv8s(torch.cat([fm, x8s], 1))

        fm = self.up8sTo4s(fm)
        fm = self.conv4s(torch.cat([fm, x4s], 1))

        fm = self.up4sTo2s(fm)
        fm = self.conv2s(torch.cat([fm, x2s], 1))

        fm = self.up2sToraw(fm)
        x = self.convraw(torch.cat([fm, x], 1))
        seg_pred = x[:, : self.seg_dim, :, :]
        ver_pred = x[:, self.seg_dim :, :, :]

        return seg_pred, ver_pred


if __name__ == "__main__":
    # test varying input size
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    for k in range(50):
        # hi, wi = np.random.randint(0, 29), np.random.randint(0, 49)
        # h, w = 256 + hi * 8, 256 + wi * 8
        h, w = 256, 256
        print('input: (h, w) = {}, {}'.format(h, w))
        img = np.random.uniform(-1, 1, [1, 3, h, w]).astype(np.float32)
        net = ResNet18_8s(1, 1).cuda()
        seg_pred, ver_pred = net(torch.tensor(img).cuda())

        print("output: seg_pred = {}".format(seg_pred))

        # tensor -> numpy
        seg_pred = seg_pred.to('cpu').detach().numpy().copy()
        seg_pred = np.resize(seg_pred, [seg_pred.shape[3], seg_pred.shape[2], seg_pred.shape[1]])

        fig = plt.figure()
        plt.imshow(seg_pred)
        plt.show()

        # ---------------
        # 終了処理
        # ---------------
        # Troch が占有している GPU メモリを開放する
        # https://base64.work/so/python/592952
        del img, net, seg_pred, ver_pred
        torch.cuda.empty_cache()
