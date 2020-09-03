import cv2
import numpy as np


def sift(src):
    new_img = src.copy()
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(new_img, None)
    dst = cv2.drawKeypoints(new_img, keypoints, None, flags=4)

    return dst


def 

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
    # エッジ抽出
    # -----------------
    src = sift(img)

    # 画像をarrayに変換
    im_list = np.asarray(src)
    # 貼り付け
    plt.imshow(im_list, cmap="gray")
    # 表示
    plt.show()
