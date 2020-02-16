import os
import glob
import cv2
import numpy as np


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


#----------------------
# 画像をタイル状に連結
#----------------------
def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])


#----------------------
# 変数
#----------------------
im_list = []
im_width = 120
im_height = 120
output_name = 'concat_tile.bmp'
tile_range = 5
#----------------------
# Path取得
#----------------------
f_path = os.getcwd()
input_path = f_path + '\\*.bmp'
output_path = f_path + '\\' + str(output_name)
f_list = glob.glob(input_path) # カレントディレクトリ内の画像の全てのパスを取得: list

rem = len(f_list) % tile_range



src_list = []
list_len = tile_range
for i, f in enumerate(f_list):
    src = cv2.imread(f)
    src = scale_box(src, im_width, im_height)
    src_shape = src.shape
    src_list.append(src)
    i += 1
    if i % tile_range == 0:
        im_list.append(src_list)
        src_list = []

if len(src_list) != list_len:  # tileの列数より画像枚数が少ないとき
    loop = list_len - len(src_list)
    for j in range(loop):
        array = np.zeros(src_shape, dtype=np.uint8)
        src_list.append(array)
    im_list.append(src_list)

im_tile = concat_tile(im_list)
#cv2.imshow('tile', im_tile)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite(output_path, im_tile, im_tile)
