import os
import cv2
import numpy as np
from glob import glob


class DataLoad:
    def __init__(self, dir_path, extension='png', size=None, channel=3, framework='keras'):
        self.opt = {
            'dir_path': None,
            'extension': None,
            'resize': True,
            'width': None,
            'height': None,
            'channel': None,
            'framework': None
        }

        # sizeが決定されているとき
        if size != None:
            self.validate_img_size(size)
            self.opt['width'], self.opt['height'] = size[0], size[1]
        else:
            self.opt['resize'] = False

        self.validate_dir_path(dir_path)
        self.validate_extension(extension)
        self.validate_channel(channel)
        self.validate_framework(framework)
        self.opt['dir_path'] = dir_path
        self.opt['extension'] = extension
        self.opt['channel'] = channel
        self.opt['framework'] = framework

    def data_load(self):
        img_data = []
        lbl_data = []

        # ラベルとなるディレクトリ名を取得
        d_pathes = glob(self.opt['dir_path']+os.sep+'*')
        self.validate_d_pathes(d_pathes)
        num_classes = len(d_pathes)
        CLS = [os.path.basename(p) for p in d_pathes]

        ######################
        #  データをロードする  #
        ######################
        for parent_dir_path in d_pathes:
            child_dir_path = [p for p in
                              glob(parent_dir_path+os.sep+'**'+os.sep+'*.' + self.opt['extension'],
                                   recursive=True)
                              if os.path.isfile(p)]
            if child_dir_path:  # もしdata配列に値がある場合
                for i in child_dir_path:
                    ##################
                    #  画像を読み込む  #
                    ##################
                    img = cv2.imread(i)
                    self.validate_img(img)  # 画像がロードできなかった場合
                    img = self.img_convert(img)
                    img_data.append(img)

                    ##################
                    # ラベルを作成する #
                    ##################
                    Label = parent_dir_path[parent_dir_path.find(
                        os.sep):].strip(os.sep)
                    # 使用するフレームワークによって，正解ラベルの作成方法が異なる
                    if (self.opt['framework'] is 'Tensorflow') or (self.opt['framework'] is 'keras'):
                        # one-hot-labelを作成する
                        v = [1 if cls == str(
                            Label) else 0 for i, cls in enumerate(CLS)]
                        # one-hot-labelの要素の合計が 1 か確認
                        self.validate_OneHotLabel(v)
                    elif (self.opt['framework'] is 'PyTorch'):
                        v = float(Label)

                    lbl_data.append(v)

                    ###########################
                    #  読み込んだ画像情報を表示  #
                    ###########################
                    img_name = i[i.find(os.sep):].strip(os.sep)
                    print(
                        f'path : {img_name}, size : ({img.shape[0]}, {img.shape[1]}, {img.shape[2]}), label : {Label}')

        img_data = np.array(img_data, dtype=np.float32)
        lbl_data = np.array(lbl_data, dtype=np.int)
        # img_dataとlbl_dataの要素数が同等か判定
        self.validate_img_label(img_data, lbl_data)
        print(
            f'Number of read images : {len(img_data)}, Categories : {num_classes}')

        return img_data, lbl_data

    def img_convert(self, img):
        if self.opt['resize']:
            img = cv2.resize(
                img, (self.opt['width'], self.opt['height'])).astype(np.float32)
            img /= 255.

        return img

    #########################
    #  初期設定で確認する項目  #
    #########################
    def validate_dir_path(self, dir_path):
        assert str(dir_path)

    def validate_extension(self, extension):
        assert str(extension)

    def validate_img_size(self, size):
        assert tuple(size)
        assert int(size[0])
        assert int(size[1])

    def validate_channel(self, channel):
        assert 2 <= int(channel) <= 4

    def validate_framework(self, framework):
        assert str(framework)

    ######################
    #  関数内での確認事項  #
    ######################
    def validate_d_pathes(self, d_pathes):
        """d_pathesが空のリストならエラー"""
        assert d_pathes

    def validate_img(self, img):
        """imreadでイメージが読み込めなかったときエラー"""
        assert len(img) != 0

    def validate_OneHotLabel(self, lbl):
        """one-hot-labelとして作成された配列の要素の合計が1にならない場合エラー"""
        assert np.sum(lbl) == 1

    def validate_img_label(self, img_data, lbl_data):
        """imgデータ配列とLabelデータ配列の大きさが同等でない場合にエラー"""
        assert len(img_data) == len(lbl_data)


if __name__ == "__main__":
    print(os.getcwd())
    DirPath = '../DataSet/AngleDetection/training'
    img_width, img_height = 64, 64
    #DirPath = 'D:\My_programing\python\AI\DataSet\AngleDetection\training'
    #DirPath = "D:\My_programing\python\AI\Training"
    load = DataLoad(DirPath, 'jpg', size=(img_width, img_height))
    img, lbl = load.data_load()
    print(lbl)
