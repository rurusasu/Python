import torch
from torch import nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=middle_channels,
                        kernel_size=3,
                        padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        