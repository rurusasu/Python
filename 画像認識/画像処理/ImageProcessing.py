from PIL import Image
import cv2
import numpy as np


#class ImageProcessingError(Exception):

#--------------------------------------
# アスペクト比を固定してリサイズする関数
#--------------------------------------
def scale_box(src, width, height):
    """
    アスペクト比を固定して、指定した大きさに収まるようリサイズする。

    Parameters
    ----------
    src : OpenCV型
        入力画像
    width : int
        変換後の画像幅
    height : int
        変換後の画像高さ
    
    Return
    ------
    dst : OpenCV型
    """
    scale = max(width / src.shape[1], height / src.shape[0])
    return cv2.resize(src, dsize=None, fx=scale, fy=scale)



def LoadImage(ImageWidth, ImageHeight, Camera_num=None, ImagePath=None):
    if ImageWidth <= 0 or ImageHeight <= 0:
        raise ValueError('入力画像の幅と高さは0より大きい必要があります。')
    if Camera_num != None and ImagePath != None:
        raise ValueError('カメラを起動するか画像を読み込むかのどちらかのみを使ってください。')
    if Camera_num == None and ImagePath == None:
        raise ValueError('カメラを起動するか画像を読み込むかのどちらかを指定してください。')
    
    #---------------------
    # カメラを起動する
    #---------------------
    if Camera_num != None:
        if type(Camera_num) is not int:
            raise ValueError('Camerea_numは整数で指定してください。')
        # Setup the camera as a capture device
        cap = cv2.VideoCapture(Camera_num)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ImageWidth)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ImageHeight)
        _, frame = cap.read()
        if not cap.isOpened(): return None
    #---------------------
    # 画像を読み込む
    #---------------------
    elif ImagePath != None:
        if type(ImagePath) is not str:
            raise ValueError('ファイルパスは文字列で指定する必要があります。')
        
        # 画像を読み込む
        frame = cv2.imread(ImagePath)
        if frame is None:
            raise ValueError('画像を読み込めませんでした。')
        frame = scale_box(frame, ImageWidth, ImageHeight)
        #cv2.imshow("color", frame)
    return frame


#----------------------------------
# 画像を要素毎に分解
#----------------------------------
def Disassembly(src):
    """
    画像を3つの要素に分解

    Parameter
    ---------
    src : OpenCV型
        分解前の画像
            
    Return
    ------
    dst : list
        要素のリスト
    """
    new_src = src.copy() # 画像をコピー

    if new_src.ndim == 2: # グレースケール
        dst = new_src
    elif new_src.ndim == 3:
        src_1, src_2, src_3 = cv2.split(new_src)  # Original → R,G,B(or H,S,V)
        dst = [src_1, src_2, src_3]
    elif new_src.ndim == 4:
        src_1, src_2, src_3, src_4 = cv2.split(new_src)
        dst = [src_1, src_2, src_3, src_4]
    return dst


#----------------------------------
# 二値化方法
#----------------------------------
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
    new_src = src.copy()
    ret, dst = cv2.threshold(new_src, threshold, 255, Type)
    return ret, dst


def otsu_binalize(gray, min_value=0, max_value=255):
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
    if gray.ndim != 2:
        gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

    # ヒストグラムの算出
    hist = [np.sum(gray == i) for i in range(256)]
    s_max = (0, -10)

    for th in range(256):
        # クラス1とクラス2の画素数を計算
        n1 = sum(hist[:th])
        n2 = sum(hist[th:])

        # クラス1とクラス2の画素値の平均を計算
        if n1 == 0: mu1 = 0
        else: mu1 = sum([i * hist[i] for i in range(0, th)]) / n1
        if n2 == 0: mu2 = 0
        else: mu2 = sum([i * hist[i] for i in range(th, 256)]) / n2

        # クラス間分散の分子を計算
        s = n1*n2*(mu1-mu2)**2

        # クラス間分散の分子が最大のとき、クラス間分散の分子と閾値を記録
        if s > s_max[1]:
            s_max = (th, s)
    
    # クラス間分散が最大のとき閾値を取得
    t = s_max[0]

    # 算出した閾値で二値化処理
    gray[gray < t] = min_value
    gray[gray >= t] = max_value

    return gray




if __name__ == "__main__":
    import sys, os
    sys.path.append(os.getcwd())
    #ImagePath = 'D:/myfile/My_programing/python/画像認識/画像処理/resource/lena.png'
    ImagePath = r'D:\myfile\My_programing\python\Data\AngleDetection\DataSet\ALL_0312\training\0\image1.jpg'
    src = LoadImage(299, 299, ImagePath=ImagePath) # 画像を読み込む

    #dst = Disassembly(src)
    dst = otsu_binalize(src)
    cv2.imshow('binarize', dst)
    cv2.waitKey(0)