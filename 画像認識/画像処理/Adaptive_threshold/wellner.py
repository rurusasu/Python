from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())

class IndivisibleError(Exception):
    """
    sの値が割り切れない場合に発生させたい例外
    """
    pass


def main():
    # オリジナル画像の読み込み
    with Image.open('resource/lena.png') as org_img:
        # グレースケール化
        gray = org_img.convert('L')
        gray_array = np.asarray(gray, np.uint8)
    #print(gray_array.shape)
    # 画像の高さ(height), 幅(width), チャネル数(channel)を取得
    (height, width) = gray_array.shape

    # 画素値を1行に整列
    gray_array = gray_array.reshape((-1, 1))
    try:
        # もし画像の幅が8で割り切れない場合
        if width % 8 != 0: raise IndivisibleError() # 例外を発生させる
        else: s = int(width / 8)
    except Exception:
        print('割り切れないピクセル数です')
        return

    t = 15
    output = np.empty(0)
    #移動平均(Moving average)を計算するためのlistを作成
    MA_list = np.zeros((s))

    for i, v in enumerate(gray_array):
        # list内の移動平均を計算する
        MA = MA_list.sum() / s
        MA_list_2 = MA_list[:-1].copy()
        MA_list = np.insert(MA_list_2, 0, v)
        #---------------------
        # 二値化処理する
        #---------------------
        if v < MA * ((100-t) / 100):
            v = 255
        else:
            v = 0
        
        v = np.array(v)
        output = np.r_[output, v]
    print('処理が終了しました。')
    output = output.reshape((height, width)).astype(np.uint8)
    
    # 画像を表示
    plt.imshow(output, vmin=0, vmax=255, interpolation='none')
    plt.imshow(output)
    
    Image.fromarray(output).save('resource/lena_2.png', quality=100)
    #Image.save('resource/lena_2.jpg', quality=95)


if __name__ == "__main__":
    main()
