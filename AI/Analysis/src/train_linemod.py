

def train(net, optimizer, dataloader, epoch):
    """dataloader で読みだされたデータを用いてネットを訓練する関数

    Args:
        net ([type]): [description]
        optimizer ([type]): [description]
        dataloader ([type]): [description]
        epoch ([type]): [description]
    """
    for rec in recs.reset()