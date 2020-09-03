import cv2


def laplacian(src, bit=cv2.CV_64F, ksize=3):
    """
    ラプラシアンフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    bit
        出力画像のビット深度
    ksize
        カーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.Laplacian(new_img, bit, ksize)

    return dst


def prewitt(src, dx=1, dy=1):
    """
    プレヴィットフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    dx
        x軸方向微分の次数
    dy
        y軸方向微分の次数
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出
    
    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    if dx == 1 and dy == 0:
        dst = cv2.filter2D(new_img, -1, kernelx)
    elif dx == 0 and dy == 1:
        dst = cv2.filter2D(new_img, -1, kernely)
    elif dx == 1 and dy == 1:
        dst_x = cv2.filter2D(new_img, -1, kernelx)
        dst_y = cv2.filter2D(new_img, -1, kernely)
        dst = dst_x + dst_y
    else:
        print("dx, dy は 0 もしくは 1 を指定してください。")
        dst = None

    return dst


def sobel(src, bit=cv2.CV_64F, dx=1, dy=1, ksize=3):
    """
    ソーベルフィルタによる空間フィルタリングを行う関数

    Parameters
    ----------
    src : OpenCV型
        入力画像
    bit
        出力画像のビット深度
    dx
        x軸方向微分の次数
    dy
        y軸方向微分の次数
    ksize
        カーネルサイズ

    (dx, dy) = (1, 0) : 横方向の輪郭検出

    (dx, dy) = (1, 0) : 縦方向の輪郭検出
    
    (dx, dy) = (1, 1) : 斜め右上方向の輪郭検出
    
    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.Sobel(img, bit, dx, dy, ksize)

    return dst


def canny(src, thresh1=100, thresh2=200):
    """
    Cannyアルゴリズムによって、画像から輪郭を取り出すアルゴリズム

    Parameters
    ----------
    src : OpenCV型
        入力画像
    thresh1
        最小閾値(Hysteresis Thresholding処理で使用)
    thresh2
        最大閾値(Hysteresis Thresholding所rで使用)

    Returns
    -------
    dst : OpenCV型
        出力画像

    """
    new_img = src.copy()
    dst = cv2.Canny(new_img, thresh1, thresh2)

    return dst


def LoG(src, ksize=(3, 3), sigmaX=1.3, l_ksize=3):
    """
    ガウシアンフィルタで画像を平滑化してノイズを除去した後、ラプラシアンフィルタで輪郭を取り出す

    Parameters
    ----------
    src : OpenCV型
        入力画像
    ksize : tuple
        ガウシアンフィルタのカーネルサイズ
    sigmaX
        ガウス分布のσ
    l_ksize
        ラプラシアンフィルタのカーネルサイズ

    Returns
    -------
    dst : OpenCV型
        出力画像
    """
    new_img = src.copy()
    dst = cv2.GaussianBlur(new_img, ksize, sigmaX)
    dst = laplacian(dst, ksize=l_ksize)

    return dst


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np

    img_path = "./data/lena.png"
    #img = cv2.imread(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Image is not read.")
    else:
        print("Image is read.")
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # print(img.shape)
    # -----------------
    # エッジ抽出
    # -----------------
    # src = laplacian(img)
    # src = laplacian(img, ksize=5)

    # src = prewitt(img)

    # src = sobel(img)
    # src = sobel(img, dx=0)
    src = sobel(img, dy=0)
    # src = sobel(img, dx=0, ksize=5)

    # src = canny(img)

    # src = LoG(img)

    # 画像をarrayに変換
    im_list = np.asarray(src)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    # 表示
    plt.show()
