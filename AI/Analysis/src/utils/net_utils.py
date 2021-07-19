from _typeshed import ReadableBuffer
import os
import torch


def load_model(model, optim, model_dir: str, epoch: int = -1) -> int:
    """
    訓練途中に保存されたモデルが `model_dir` に `.pth`として存在する場合，そのモデルの (`parameter`, `optimizer`) を引数 (`model`, `optim`) にコピーし，`epoch + 1` を返す関数

    Args:
        model (): パラメタがコピーされるモデルの器
        optim ([type]): 最適化関数がコピーされるモデルの器
        model_dir (str): 訓練途中に保存されるモデルの保存先のパス
        epoch (int, optional): 何 epoch 目に保存されたモデルをロードするか. `-1` の場合，最後に保存されたモデルをロードする．Defaults to -1.

    Returns:
        int: 保存されたモデルの epoch 数 +1
    """
    if not os.path.exist(model_dir):  # ディレクトリが存在しない場合
        return 0

    pths = [int(pth.split(".")[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    # 事前に保存された訓練済みモデルが存在する場合
    pretrained_model = torch.load(os.path.join(model_dir, "{}.pth".format(pth)))
    model.load_state_dict(pretrained_model["net"])
    optim.load_state_dict(pretrained_model["optim"])
    print("load model {} epoch {}".format(model_dir, pretrained_model["epoch"]))
    return pretrained_model["epoch"] + 1


def compute_precision_recall(scores, target, reduce=False):
    """
    1つのカテゴリに対する `precision` と `recall` それぞれを返す関数．

    Args:
        scores ([type]): [description]
        target ([type]): [description]
        reduce (bool, optional): minibatch ごとに平均化するか. Defaults to False.

    Returns:
        float: 1カテゴリに対する `precision` と `recall`.
    """
    (b,) = scores.shape[0]
    preds = torch.argmax(scores, 1)
    preds = preds.float()
    target = target.float()

    tp = preds * target
    fp = preds(1 - target)
    fn = (1 - preds) * target

    tp = torch.sum(tp.view(b, -1), 1)
    fn = torch.sum(fn.view(b, -1), 1)
    fp = torch.sum(fp.view(b, -1), 1)

    precision = (tp + 1) / (tp + fp + 1)
    recall = (tp + 1) / (tp + fn + 1)

    if reduce:
        precision, recall = torch.mean(precision), torch.mean(recall)
    return precision, recall