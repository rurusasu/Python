from NuralNet_App import NuralNet_APP

class NuralNetApp():
    def __init__(self):
        self.App = NuralNet_APP()

    def Train_Input(self, Train_Data_Path, Train_Label_Path, batch_size, epochs, feature=None, val_Data_Path=None, val_Label_Path=None):
        """
        APIを用いて、自作のNuralNetに学習用データを送る関数

        Parameters
        ----------
        Train_Data_Path : str
            学習用データのファイルパス
        Train_Label_Path : str
            学習用データラベルのファイルパス
        batch_size : int
            バッチサイズ
        epochs : int
            エポック数
        feature : int(0~2) or None
            0 : 標準化
            1 : 正規化
            2 : 標準化と正規化の両方
            None : 何も行わない
        val_Data_Path : str
            検証用データのファイルパス
        val_Label_Path : str
            検証用データラベルのファイルパス
        """
        Train_Path = (Train_Data_Path, Train_Label_Path)
        if val_Data_Path is None or val_Label_Path is None:
            Val_Path = None
            print('Validationにはトレーニングデータの一部を使用します。')
        else:
            Val_Path = (val_Data_Path, val_Label_Path)
        self.App.Training_click(Train_Path, batch_size, epochs, feature, Val_Path)

    def NetFlow(self, Flow_Data):
        """
        学習したNuralNetにデータを渡し、その出力を返す関数
        ※この関数で入力したデータは学習には使用されません。

        Parameters
        ----------
        Flow_Data : list
            学習したNuralNetに通すデータ配列
        
        """
        y = self.App.Flow_click(Flow_Data)
        return y