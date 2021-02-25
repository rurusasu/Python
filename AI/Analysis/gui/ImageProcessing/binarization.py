import sys, os

sys.path.append(".")
sys.path.append("..")

import cv2


def GlobalThreshold(img, gaussian=False, threshold=127, Type=cv2.THRESH_BINARY):
    """
    画素値が閾値より大きければある値(白色'255')を割り当て，そうでなければ別の値(黒色)を割り当てる。
    img is NoneならNoneを、変換に成功すれば閾値処理された2値画像を返す。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。
    threshold : flaot
        2値化するときの閾値
    Type
        閾値の処理方法
        ・cv2.THRESH_BINARY
        ・cv2.THRESH_BINARY_INV
        ・cv2.THRESH_TRUNC
        ・cv2.THRESH_TOZERO
        ・cv2.THRESH_TOZERO_INV

    Returns
    -------
    img : OpenCV型
        変換後の画像データ
    response : int
        0: 変換できました。
        6: 画像の元データが存在しません。
    """
    response = 6
    if img is None:  # 画像の元データが存在しない場合
        return response, None
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)

    ret, img = cv2.threshold(img, threshold, 255, Type)
    response = 0
    return response, img


def OtsuThreshold(img, gaussian=False):
    """
    入力画像が bimodal image (ヒストグラムが双峰性を持つような画像)であることを仮定すると、
    そのような画像に対して、二つのピークの間の値を閾値として選べば良いと考えることであろう。これが大津の二値化の手法である。
    双峰性を持たないヒストグラムを持つ画像に対しては良い結果が得られないことになる。

    Parameters
    ----------
    img : OpenCV型
        変換前の画像データ
    gaussian : True or False
        ガウシアンフィルタを適応するか選択できる。

    Returns
    -------
    img : OpenCV型
        変換後の画像データ
    response : int
        0: 変換できました。
        6: 画像の元データが存在しません。
    """
    response = 6
    if img is None:  # 画像の元データが存在しない場合
        return response, None
    # 画像のチャンネル数が2より大きい場合
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ガウシアンフィルタで前処理を行う場合
    if gaussian:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    response = 0

    return response, img

if __name__ == "__main__":
