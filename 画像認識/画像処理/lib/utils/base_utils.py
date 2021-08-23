import os
import shutil
from typing import Type, Union
import cv2
import numpy as np


def Disassembly(src: np.ndarray) -> dict:
    """
    画像を画素ごとに要素に分解する関数

    Args:
        src (ndarray): 入力画像

    Return:
        dst_elem (list): 要素ごとの値を持つリスト．[r, g, b]．
    """
    if len(src.shape) == 3:
        r, g, b = cv2.split(src)  # Original → R,G,B(or H,S,V)
    else:
        raise ValueError("Input value is invalid.")
    # dst = [src_1, src_2, src_3]
    return [r, g, b]


def makedir(filepth: str):
    """
    ディレクトリが存在しない場合に、深い階層のディレクトリまで再帰的に作成する関数。もし、ディレクトリが存在する場合は一度中身ごと削除してから再度作成。

    Arg:
        filepth (str): 作成するディレクトリのパス
    """
    if not os.path.isdir(filepth):
        os.makedirs(filepth, exist_ok=True)
    else:
        shutil.rmtree(filepth)
        os.makedirs(filepth, exist_ok=True)


def resize(src: np.ndarray, size: Type[Union[int, list, tuple]]):
    """画像をリサイズする関数

    Args:
        src (np.ndarray): 入力画像
        size (Type[Union[int, list, tuple]]): [description]

    Return:
        dst (np.ndarray): 出力画像
    """
    dst = cv2.resize(src, size)
    return dst


def scale_box(src, width, height):
    """
    アスペクト比を固定して、指定した大きさに収まるようリサイズする。

    Args:
        src (np.ndarray): 入力画像
        width (int): 変換後の画像幅
        height (int): 変換後の画像高さ

    Return:
        dst (np.ndarray): リサイズ後の画像
    """
    scale = max(width / src.shape[1], height / src.shape[0])
    return cv2.resize(src, dsize=None, fx=scale, fy=scale)


def save_img(
    src: np.ndarray,
    filepth: str,
    filename: str,
    ext: str = ".png",
):
    img_name = filename + ext
    pth = os.path.join(filepth, img_name)
    cv2.imwrite(
        pth,
        src,
    )
