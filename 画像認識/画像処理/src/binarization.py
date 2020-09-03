import cv2

# ---------------
# グローバル二値化
# ---------------
def binalize(src, threshold=127, Type=cv2.THRESH_BINARY):
    """
    単純閾値処理を行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    threshold : int
        閾値

    Returns
    -------
    ret : int
        入力した閾値と同値

    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, threshold, 255, Type)
    return ret, dst


# ---------------
# 大津の二値化
# ---------------
def otsu_binalize(src):
    """
    大津の閾値処理を行う関数

    Parameter
    ---------
    new_img : OpenCV型
        入力画像
    
    Returns
    -------
    ret : int
        計算した閾値
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    ret, dst = cv2.threshold(new_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret, dst


# ---------------
# 適応的二値化
# ---------------
def adaptive_binalize(
    src, method=cv2.ADAPTIVE_THRESH_MEAN_C, Type=cv2.THRESH_BINARY, block_size=11, C=2
):
    """
    適応的閾値処理を行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    """
    new_img = src.copy()
    dst = cv2.adaptiveThreshold(new_img, 255, method, Type, block_size, C)
    return dst


if __name__ == "__main__":
    # import sys
    # import os
    # print(os.getcwd())
    # sys.path.append('.')
    # sys.path.append('..')

    from matplotlib import pyplot as plt
    import numpy as np

    img_path = "./data/lena.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image is not read.")
    else:
        print("Image is read.")

    # print(img.shape)
    # -----------------
    # 二値化テスト
    # -----------------
    _, src = binalize(img)
    # _, src = otsu_binalize(img)
    # src = adaptive_binalize(img)

    # 画像をarrayに変換
    im_list = np.asarray(src)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    # 表示
    plt.show()

