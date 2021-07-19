import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../../")

import torch
import torchvision.utils as vutils
from easydict import EasyDict
from torch.tensor import Tensor

from tensorboardX import SummaryWriter


class History:
    def load_dict(self, *args):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()

    def update(self):
        raise NotImplementedError()


class LossHistory(History):
    def __init__(self) -> None:
        self.losses = {"train": [], "dev": []}
        self.accs = {"train": [], "dev": []}
        self.bounding_accs = {"train": [], "dev": []}
        self.shrink = {"train": [], "dev": []}

    def load_dict(self, other):
        self.losses = other.losses
        self.accs = other.accs
        self.bounding_accs = other.bounding_accs
        self.shrink = other.shrink


class AverageMeter(EasyDict):
    """Computes and stores the average and current value
    """

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    colors = [
        [0, 0, 0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(self, rec=True, rec_dir=None, dump_fn=None):
        from matplotlib import cm

        if rec:
            self.writer = SummaryWriter(log_dir=rec_dir)
            self.cmap = cm.get_cmap()
        else:
            self.writer = None

        self.dump_fn = dump_fn

    def rec_loss(self, loss, step, name="data/loss"):
        msg = "{} {} {}".format(name, step, loss)
        print(msg)
        if self.dump_fn is not None:
            with open(self.dump_fn, "a") as f:
                f.write(msg + "\n")

        if self.writer is None:
            return

        self.writer.add_scalar(name, loss, step)

    def rec_loss_batch(self, losses_batch, step, epoch, prefix="train"):
        msg = "{} epoch {} step {}".format(prefix, epoch, step)
        for k, v in losses_batch.items():
            msg += " {} {:.8f} ".format(k.split("/")[-1], v)

        print(msg)
        if self.dump_fn is not None:
            with open(self.dump_fn, "a") as f:
                f.write(msg + "\n")

        if self.writer is None:
            return

        for k, v in losses_batch.items():
            self.writer.add_scalar(k, v, step)

    def rec_segmentation(self, seg, num_classes, nrow, step, name="seg"):
        if self.writer is None:
            return

        seg = torch.argmax(seg, dim=1).long()
        r = seg.clone()
        g = seg.clone()
        b = seg.clone()
        for l in range(num_classes):
            inds = seg == l
            r[inds] = self.colors[l][0]
            g[inds] = self.colors[l][1]
            b[inds] = self.colors[l][2]
        seg = torch.stack([r, g, b], dim=1)

        seg = vutils.make_grid(seg, nrow)
        self.writer.add_image(name, seg, step)

    def rec_vertex(self, vertex, mask, nrow, step, name="vertex"):
        if self.writer is None:
            return

        vertex = (vertex[:, :2, ...] * mask + 1) / 2
        height, width = vertex.shape[2:]
        vertex = vertex.view(-1, height, width)
        vertex = self.cmap(vertex.detach().cpu().numpy())[..., :3]
        vertex = vutils.make_grid(torch.from_numpy(vertex).permute(0, 3, 1, 2), nrow)
        self.writer.add_image(name, vertex, step)


def adjust_learning_rate(optimizer, epoch, lr_decay_rate, lr_decay_epoch, min_lr=1e-5):
    if ((epoch + 1) % lr_decay_epoch) != 0:
        return

    for param_group in optimizer.param_groups:
        # print(param_group)
        lr_before = param_group["lr"]
        param_group["lr"] = param_group["lr"] * lr_decay_rate
        param_group["lr"] = max(param_group["lr"], min_lr)
    print(
        "changing learning rate {:5f} to {:.5f}".format(
            lr_before, max(param_group["lr"], min_lr)
        )
    )


def compute_precision_recall(scores: Tensor, target: Tensor, reduce: bool = False):
    """precision と recall を計算する関数

    Args:
        scores (Tensor): ネットワークからの出力
        target (Tensor): 真値
        reduce (bool, optional): [description]. Defaults to False.
    """
    b = scores.shape[0]
    preds = torch.argmax(scores, 1)
    preds = preds.float()
    target = target.float()

    tp = preds * target
    fp = preds * (1 - target)
    fn = (1 - preds) * target

    tp = torch.sum(tp.view(b, -1), 1)
    fn = torch.sum(fn.view(b, -1), 1)
    fp = torch.sum(fp.view(b, -1), 1)

    precision = (tp + 1) / (tp + fp + 1)
    recall = (tp + 1) / (tp + fn + 1)

    if reduce:
        precision, recall = torch.mean(precision), torch.mean(recall)
    return precision, recall


def load_model(model, optim, model_dir, epoch=-1):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    model.load_state_dict(pretrained_model["net"])
    optim.load_state_dict(pretrained_model["optim"])
    print("load model {} epoch {}".format(model_dir, pretrained_model["epoch"]))
    return pretrained_model["epoch"] + 1


def save_model(net, optim, epoch, model_dir):
    os.system("mkdir -p {}".format(model_dir))
    torch.save(
        {"net": net.state_dict(), "optim": optim.state_dict(), "epoch": epoch},
        os.path.join(model_dir, "{}.pth".format(epoch)),
    )

