import os
import sys

sys.path.append(".")
sys.path.append("..")
import datetime
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import archs
import losses
from arg_utils import parse_args
from dataset import Dataset
from metrics import iou_score
from src.config.config import cfg
from utils import AverageMeter, makedir


def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        input = input.cuda()  # 入力画像
        target = target.cuda()  # マスク画像

        # compute output
        if config["deep_supervision"]:
            optputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters["loss"].update(loss.item(), input.size(0))
        avg_meters["iou"].update(iou, input.size(0))

        postfix = OrderedDict(
            [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg),]
        )
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
    )


def validate(config, val_loader, model, criterion):
    avg_meters = {"loss": AverageMeter(), "iou": AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config["deep_supervision"]:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)

            avg_meters["loss"].update(loss.item(), input.size(0))
            avg_meters["iou"].update(iou, input.size(0))

            postfix = OrderedDict(
                [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg),]
            )

            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict(
        [("loss", avg_meters["loss"].avg), ("iou", avg_meters["iou"].avg)]
    )


def main():
    config = vars(parse_args())
    now = datetime.datetime.now()

    if config["name"] is None:
        if config["deep_supervision"]:
            config["name"] = "%s_%s_wDS_%s" % (
                config["dataset"],
                config["arch"],
                now.strftime("%Y%m%d_%H%M%S"),
            )
        else:
            config["name"] = "%s_%s_woDS_%s" % (
                config["dataset"],
                config["arch"],
                now.strftime("%Y%m%d_%H%M%S"),
            )
    output_path = os.path.join(cfg.UNET_RESULTS_DIR, config["name"])
    try:
        os.makedirs(output_path, exist_ok=True)
    except Exception as e:
        print(e)

    models_path = os.path.join(output_path, "models")
    os.mkdir(models_path)

    with open(os.path.join(models_path, "config.yml"), "w") as f:
        yaml.dump(config, f)

    print("-" * 20)
    for key in config:
        print("%s: %s" % (key, config[key]))
    print("-" * 20)

    # Tensorboad 用のログを記録するディレクトリパス
    log_dir = os.path.join(output_path, "log")
    os.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # define loss function(criterion)
    if config["loss"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config["loss"]]().cuda()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config["arch"])
    model = archs.__dict__[config["arch"]](
        config["num_classes"], config["input_channels"], config["deep_supervision"]
    )

    model = model.cuda()

    # モデルを TensorBorad で表示するため，ログに保存
    # image = torch.randn(1, 3, 2224, 224)
    # writer.add_graph(model, image)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config["optimizer"] == "Adam":
        optimizer = optim.Adam(
            params, lr=config["lr"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "SGD":
        optimizer = optim.SGD(
            params,
            lr=config["lr"],
            momentum=config["momentum"],
            nesterov=config["nesterov"],
            weight_decay=config["weight_decay"],
        )
    else:
        raise NotImplementedError

    # scheduler
    if config["scheduler"] == "CosineAnnealingLR":
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=config["epochs"], eta_min=config["min_lr"]
        )
    elif config["scheduler"] == "ReduceLROnPlateau":
        scheduler = lr_scheduler(
            optimizer=optimizer,
            factor=config["factor"],
            patience=config["patience"],
            verbose=1,
            min_lr=config["min_lr"],
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=[int(e) for e in config["milestones"].split(",")],
            gamma=config["gamma"],
        )
    elif config["scheduler"] == "ConstantLR":
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    if config["dataset"] == "dsb2018_96":
        input_dir = cfg.DSB2018_96_DIR
        img_ids = glob(os.path.join(input_dir, "images", "*" + config["img_ext"]))
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

        train_img_ids, val_img_ids = train_test_split(
            img_ids, test_size=0.2, random_state=41
        )

        train_transform = Compose(
            [
                transforms.RandomRotate90(),
                transforms.Flip(),
                OneOf(
                    [
                        transforms.HueSaturationValue(),
                        transforms.RandomBrightness(),
                        transforms.RandomContrast(),
                    ],
                    p=1,
                ),
                transforms.Resize(config["input_h"], config["input_w"]),
                transforms.Normalize(),
            ]
        )

        val_transform = Compose(
            [
                transforms.Resize(config["input_h"], config["input_w"]),
                transforms.Normalize(),
            ]
        )

        train_dataset = Dataset(
            img_ids=train_img_ids,
            img_dir=os.path.join(input_dir, "images"),
            mask_dir=os.path.join(input_dir, "masks"),
            img_ext=config["img_ext"],
            mask_ext=config["mask_ext"],
            num_classes=config["num_classes"],
            transform=train_transform,
        )

        val_dataset = Dataset(
            img_ids=val_img_ids,
            img_dir=os.path.join(input_dir, "images"),
            mask_dir=os.path.join(input_dir, "masks"),
            img_ext=config["img_ext"],
            mask_ext=config["mask_ext"],
            num_classes=config["num_classes"],
            transform=val_transform,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
    )

    log = OrderedDict(
        [
            ("epoch", []),
            ("lr", []),
            ("loss", []),
            ("iou", []),
            ("val_loss", []),
            ("val_iou", []),
        ]
    )

    best_iou = 0
    trigger = 0
    for epoch in range(config["epochs"] + 1):
        print("Epoch [%d/%d]" % (epoch, config["epochs"]))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config["scheduler"] == "CosineAnnealingLR":
            scheduler.step()
        elif config["scheduler"] == "ReduceLROnPlateau":
            scheduler.step(val_log["loss"])

        print(
            "loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f"
            % (train_log["loss"], train_log["iou"], val_log["loss"], val_log["iou"])
        )

        log["epoch"].append(epoch)
        log["lr"].append(config["lr"])
        log["loss"].append(train_log["loss"])
        log["iou"].append(train_log["iou"])
        log["val_loss"].append(val_log["loss"])
        log["val_iou"].append(val_log["iou"])

        # Tensorboard用のデータ
        writer.add_scalar("training loss", train_log["loss"], epoch)
        writer.add_scalar("validation loss", val_log["loss"], epoch)

        pd.DataFrame(log).to_csv("%s/log.csv" % (log_dir), index=False)

        if epoch == 0:
            best_loss = val_log["loss"]
        trigger += 1

        # Best Model Save
        # if val_log['iou'] > best_iou:
        if (val_log["iou"] > best_iou) and (val_log["loss"] <= best_loss):
            torch.save(model.state_dict(), "%s/model.pth" % (models_path))
            best_iou = val_log["iou"]
            best_loss = val_log["loss"]
            print("=> saved best model")
            trigger = 0

        # early stopping
        if (
            config["early_stopping"] >= 0 and trigger >= config["early_stopping"]
        ) or val_log["loss"] < 1e-4:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()

    # summary writer を必要としない場合，close()メソッドを呼び出す
    writer.close()


if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # config = parse_args()
    # config = vars(config)
    # s = glob(os.path.join('inputs', config['dataset'],'*', 'images', ))
    # print(s)
    main()
