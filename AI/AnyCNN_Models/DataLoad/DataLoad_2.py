import cv2
import numpy as np
from glob import glob

num_classes = 2
img_height, img_width = 64, 64

class DataAugumentation:
    def Data_load(self, DirPath, CLS = None, Framework='Tensorflow'):
        """
        ディレクトリ名を画像のラベルとしてLoadする．
        """
        xs = [] # 学習用データ
        ts = []
        paths = []

        #if CLS != None:

        # 正解ラベルを文字として取得したい場合
        #t = [i[i.find('\\'):].strip('\\') for i in glob('../../DataSet/AngleDetection/training/*')]
        #cls = dir_path_1[dir_path_1.find('\\'):].strip('\\') # dir_path_1上の文字'\\'以降の文字列をtに代入

        for dir_path_1 in glob(DirPath):
            for dir_path_2 in glob(dir_path_1 + '/*'):
                for path in glob(dir_path_2 + '/*'):
                    print(path, 'を読み込みました。')
        
                    x = cv2.imread(path)
                    x = cv2.resize(x, (img_width, img_height)).astype(np.float32)
                    x /= 255.
                    xs.append(x)
                    
                    # 正解ラベルを番号として取得したい場合
                    if (Framework is 'Tensorflow') or (Framework is 'Keras'):
                        #t = [0 for _ in range(num_classes)]
                        #for i, cls in enumerate(CLS):
                            #if cls in path:
                            #    t[i] = 1
                        t = float(dir_path_1[dir_path_1.find('\\'):].strip('\\'))
                        ts.append(t)
                    elif Framework is 'PyTorch': 
                        t = np.zeros((1))
                        #t = np.array((i))
                        #ts = np.r_[ts, t]
                    paths.append(dir_path_2)

        if Framework is 'Tensorflow':
            xs = np.array(xs, dtype=np.float32)
            ts = np.array(ts, dtype=np.int)
        #elif Framework is 'PyTorch': xs = xs.transpose(0, 3, 1, 2)

        return xs, ts, paths


    def MiniBatch(self, xs, ts, mb, mbi):
        train_ind = np.arange(len(xs)) # 学習dataのインデックス配列
        np.random.seed(0)
        np.random.shuffle(train_ind)

        if mbi + mb > len(xs):
            mb_ind = train_ind[mbi:]
            np.random.shuffle(train_ind)
            mb_ind = np.hstack((mb_ind, train_ind[:(mb-(len(xs)-mbi))]))
            mbi = mb - (len(xs) - mbi)
        else:
            mb_ind = train_ind[mbi: mbi+mb]
            mbi += mb

        x = xs[mb_ind]
        t = ts[mb_ind]
        
        return x, t


if __name__ == "__main__":
    #print(glob('../../DataSet/AngleDetection/training/*'))
    import DataLoad
    load = DataLoad.DataAugumentation()
    xs, ts, paths = load.Data_load('../../DataSet/AngleDetection/training/*')
    # xs, ts = TrainingData_load('../../DataSet/AngleDetection/training/*', 'PyTorch')
    # t = [i[i.find('\\'):].strip('\\') for i in glob('../../DataSet/AngleDetection/training/*')]
    #dir_path_1[dir_path_1.find('\\'):].strip('\\')
    print(xs)
