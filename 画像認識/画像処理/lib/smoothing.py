import cv2


def average(src, ksize=(3, 3)):
    """
    平均値フィルタを用いて画像処理を行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    ksize
        フィルタサイズ

    Return
    ------
    dst : OpenCv型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.blur(new_img, ksize=ksize)

    return dst


if __name__ == "__main__":
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
    # 平滑化フィルタテスト
    # -----------------
    # src = average(img)
    src = average(img, ksize=(5, 5))

    # 画像をarrayに変換
    im_list = np.asarray(src)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    # 表示
    plt.show()
