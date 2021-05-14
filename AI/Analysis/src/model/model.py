import torch
from torch import nn
from torch.nn import parameter

class EfficientPose(nn.Module):
    def __init__(self,
                              phi,
                              num_classes,
                              num_anchors,
                              freeze_bn = False):
        super().__init__()

        assert phi in range(7)
        self.scaled_parameters = get_scaled_parameters(phi)

        self.input_size = self.scaled_parameters["input_size"]
        self.input_sahpe = (input_size, input_size, 3)
        self.bifpn_width = self.scaled_parameters["bifpn_width"]
        self.bifpn_depth = self.scaled_parameters["bifpn_depth"]


    # input layers
    image_input =


def get_scaled_parameters(phi):
    """
    EfficientPoseを構築するために必要なすべてのスケーリングされたパラメーターを取得します
    Args:
        phi: EfficientPoseスケーリングハイパーパラメータphi

    Return:
        スケーリングされたパラメータを含む辞書
    """
    # スケーラブルなパラメータを持つ infoタプル
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    backbones = (EfficientNetB0,
                                 EfficientNetB1,
                                 EfficientNetB2,
                                 EfficientNetB3,
                                 EfficientNetB4,
                                 EfficientNetB5,
                                 EfficientNetB6)

    parameters = {"input_size": input_sizes[phi],
                                  "backbone_model": backbones[phi]}

    return parameters