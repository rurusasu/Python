import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64

def data_load(DirPath):
    """
    ディレクトリ名を画像のラベルとしてLoadする．
    """
    xs = np.ndarray((0, img_height, img_width, 3)) # 学習用データ
    ts = np.ndarray((0))

    for dir_path_1 in glob(DirPath):
        for dir_path_2 in glob(dir_path_1 + '/*'):
            for path in glob(dir_path_2 + '/*'):
                print(path, 'を読み込みました。')
                x = cv2.imread(path)
                x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
                x /= 255.
                xs = np.r_[xs, x[None, ...]]

                #t = np.zeros((1))
                # 正解ラベルを文字として取得したい場合
                t = dir_path_1[dir_path_1.find('\\'):].strip('\\') # dir_path_1上の文字'\\'以降の文字列をtに代入
                
                # 正解ラベルを番号として取得したい場合
                if '0' in path: t = np.array((0))
                elif '5' in path: t = np.array((5))
                elif '10' in path: t = np.array((10))
                elif '15' in path: t = np.array((15))
                elif '20' in path: t = np.array((20))
                elif '25' in path: t = np.array((25))
                elif '30' in path: t = np.array((30))
                elif '35' in path: t = np.array((35))
                elif '40' in path: t = np.array((40))
                elif '45' in path: t = np.array((45))
                elif '50' in path: t = np.array((50))
                elif '55' in path: t = np.array((55))
                elif '60' in path: t = np.array((60))

                ts = np.r_[ts, t]

    xs = xs.transpose(0, 3, 1, 2)

    return xs, ts


def MiniBatch(xs, ts):
    train_ind = np.arange(len(xs)) # 学習dataのインデックス配列
    np.random.seed(0)
    np.random.shuffule(train_ind)
    
    for i in range(100):
        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb


if __name__ == "__main__":
    #print(glob('../../DataSet/AngleDetection/training/*'))
    xs, ts = data_load('../../DataSet/AngleDetection/training/*')
    print(xs)
