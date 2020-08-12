# cording: utf-8

import sys, os
sys.path.append(os.getcwd())
import numpy as np
import PySimpleGUI as sg
from common.layers import Input, Dense
from common.sequential import Sequential
from common.functions import Datafeature
from common.callbacks import LearningVisualizationCallback
from common.csvIO import DataConv

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.ticker import *


def __filePath__(dir_name, file_name):
    path = str(dir_name) + '/' + str(file_name)
    return path

def dataLoad(filePath, dtype):
    if filePath != str:
        filePath = str(filePath)
    x = np.loadtxt(
        filePath,  # 読み込むファイル名(例"save_data.csv")
        dtype=dtype,  # データのtype
        delimiter=",",  # 区切り文字の指定
        ndmin=2  # 配列の最低次元
    )
    return x


# ----- The callback function ----- #
class NuralNet_APP:
    def __init__(self):
        # コンストラクタ
        self.model = Sequential()
        self.main()
        
    def DataMake_click(self, importDirPath, outputDirPath, org_FileName, lrn_FileName, tst_FileName, digit):
        """ Originalデータを学習用データと正解ラベルに分ける関数 """
        org_FileName = __filePath__(importDirPath, org_FileName)
        lrn_FileName = __filePath__(outputDirPath, lrn_FileName)
        tst_FileName = __filePath__(outputDirPath, tst_FileName)
        DataConv(org_FileName, lrn_FileName, col_range_end=2, digit=digit)
        DataConv(org_FileName, tst_FileName, col_range_first=3, col_range_end=6, digit=digit)

    def LayerAdd_click(self, layerName, Node, weight=None, bias=None, activation=None):
        """ NetWorkにレイヤを追加していく関数 """
        Node = int(Node)

        if layerName == 'input':
            self.model.add(Input(input_shape=Node))
        elif layerName == 'Dense':
            self.model.add(Dense(Node, activation, weight, bias))
        
        print(layerName + ':' + str(Node))
        return layerName


    def NetMake_click(self, loss, optimizer, metrics):
        """ 
        NetWorkに損失関数、評価関数、最適化アルゴリズムをセットし、
        モデルを構築する関数
        """
        metrics = [metrics]
        self.model.compile(loss, optimizer=optimizer, metrics=metrics)
        print('コンパイル完了')


    def Training_click(self, orgPath, batch_size, epochs, feature=None, valPath=None):
        orgLrn_Path = orgPath[0]
        orgTrg_Path = orgPath[1]

        #---------------
        # 型変換
        #---------------
        if batch_size != int:
            batch_size = int(batch_size)
        if epochs != int:
            epochs = int(epochs)
        #---------------------
        # データの読み込み
        #---------------------
        x = dataLoad(orgLrn_Path, float)
        t = dataLoad(orgTrg_Path, float)
        #-------------------------------
        # DataFeature
        #-------------------------------
        if feature != None:
            x = Datafeature(x, feature)
            t = Datafeature(t, feature)
        #-----------------------------
        # Validationデータの読み込み
        #-----------------------------
        if valPath != None:
            valRLN_Path = valPath[0]
            valTrg_Path = valPath[1]
            x_val = dataLoad(valRLN_Path, float)
            t_val = dataLoad(valTrg_Path, float)
            #-------------------------------
            # DataFeature
            #-------------------------------
            if feature != None:
                x_val = Datafeature(x_val, feature)
                t_val = Datafeature(t_val, feature)

                validation = (x_val, t_val)
        else:
            validation = None

        # 学習曲線を可視化するコールバックを用意する
        higher_better_metrics = ['r2']
        visualize_cb = LearningVisualizationCallback(higher_better_metrics)
        callbacks = [
            visualize_cb,
        ]

        self.model.fit(x=x, t=t, batch_size=batch_size, epochs=epochs, validation=validation, callbacks=callbacks)


    def Test_click(self, tstDataPath, feature):
        tstLrn_Path = tstDataPath[0]
        tstTrg_Path = tstDataPath[1]
        
        #---------------------
        # データの読み込み
        #---------------------
        x = dataLoad(tstLrn_Path, float)
        t = dataLoad(tstTrg_Path, float)  
        #-------------------------------
        # DataFeature
        #-------------------------------
        if feature != None:
            x = Datafeature(x, feature)
            t = Datafeature(t, feature)
        self.model.evaluate(x, t)

    def FlowChartPrint_click(self):
        xmin = -8
        xmax = 8
        ymin = 0
        ymax = 25
        dh = 0.7  # 行ごとの高さ
        A = []
        B = []
        for i in range(0, int(ymax//dh)+1):
            s = '{0:.1f}'.format(dh*i)
            A += [float(s)]
            B += [i]

        fig = plt.figure()
        ax1 = plt.subplot(111)
        ax1 = plt.gca()
        ax1.set_xlim([xmin, xmax])
        ax1.set_ylim([ymax, ymin])
        aspect = (abs(ymax-ymin))/(abs(xmax-xmin))*abs(ax1.get_xlim()
                                                       [1] - ax1.get_xlim()[0]) / abs(ax1.get_ylim()[1] - ax1.get_ylim()[0])

        ax1.set_aspect(aspect)

        ax1.tick_params(labelsize=6)
        ax1.xaxis.set_major_locator(MultipleLocator(1))
        ax1.yaxis.set_major_locator(MultipleLocator(dh))
        plt.yticks(A, B)
        plt.grid(which='both', lw=0.3, color='#cccccc', linestyle='-')

        # Store texts in list
        a = []
        ax = []
        ay = []
        i = 0
        iis = 3
        ii = iis+1
        wx = 10
        row = 1
        llw = 0.5
        fsize = 5  # fontsize
        title = 'model'

        # Draw title
        xs = 0
        ys = iis*dh-1.0*dh
        plt.text(xs, ys, title, rotation=0, ha='center',
                 va='center', fontsize=fsize, fontweight='bold')

        for key in self.model.sequential.keys():
            a = a+[key]
            # Store coordinats of texts in list
            ii += 1
            ys = ii*dh
            xs = 0
            ax = ax+[xs]
            ay = ay+[ys+0.5*dh]

            # Store coordinates of tbox shapes in list
            xs = -0.5*wx
            xe = 0.5*wx
            ye = ys + row*dh  # 行間

            # Draw box
            poly = Polygon(
                [(xs, ys), (xe, ys), (xe, ye), (xs, ye)],  # 左回りでプロット
                facecolor='#dddddd',
                edgecolor='#000000',
                lw=llw
            )
            ax1.add_patch(poly)

            # Draw text
            if ax[i] == 0:
                plt.text(ax[i], ay[i], a[i], rotation=0,
                         ha='center', va='center', fontsize=fsize)
            else:
                plt.text(ax[i], ay[i], a[i], rotation=0,
                         ha='left', va='center', fontsize=fsize)

            # Draw line
            ii += 1
            lx = [0, 0]
            ly = [ii*dh, (ii+1)*dh]
            plt.plot(lx, ly, 'k-', lw=0.5)
            i += 1

        plt.show()


    def Flow_click(self, flow_data):
        y = self.model.flow(flow_data)

        return y

    def main(self):
        # ----- Column Definition ----- #
        Dir = [
            [sg.Text('Import Dir')],
            [sg.Input(size=(30, 1)), sg.FolderBrowse(key='-importDirPath-')],
            [sg.Text('Output Dir')],
            [sg.Input(size=(30, 1)), sg.FolderBrowse(key='-outputDirPath-')],
            [sg.Button('DataMake', key='-DataMake-')],
        ]

        convfunc = [
            [sg.Text('OrigiData')],
            [sg.InputText('data.csv', size=(13, 1), key='-orgDataFileName-')],
            [sg.Text('LearnData')],
            [sg.InputText('learn.csv', size=(13, 1), key='-lrnDataFileName-')],
            [sg.Text('TestData'),
             sg.Text('小数点以下の桁数')],
            [sg.InputText('test.csv', size=(13, 1), key='-tstDataFileName-'),
             sg.InputText('2', size=(5, 1), key='-Digit-')],
        ]

        dataConv = [
            [sg.Column(Dir), sg.Column(convfunc)],
        ]

        NetMake = [
            [sg.Text('層の種類          '), sg.Text('ユニット数')],
            [sg.InputCombo(('input', 'Dense'), size=(15, 1), key='-LayerName-'),
             sg.InputText('50', size=(5, 1), key='-Node-')],
            [sg.Text('重みの初期値')],
            [sg.InputCombo(('He', 'Xavier',), size=(
                15, 1), key='-weightInit-')],
            [sg.Text('閾値の初期値')],
            [sg.InputCombo(('ZEROS',), size=(15, 1), key='-biasInit-')],
            [sg.Text('活性化関数')],
            [sg.InputCombo(('relu', 'sigmoid', 'liner'), size=(15, 1), key='-activation-'),
             sg.Button('add', key='-LayerAdd-')],
            [sg.Text('損失関数')],
            [sg.InputCombo(('mean_squared_error',),
                           size=(20, 1), key='-loss-')],
            [sg.Text('最適化')],
            [sg.InputCombo(('sgd',
                            'momentum_sgd',
                            'nag',
                            'ada_grad',
                            'rmsprop',
                            'ada_delta',
                            'adam',), size=(20, 1), key='-optimizer-')],
            [sg.Text('評価関数')],
            [sg.InputCombo(('r2', 'rmse'), size=(15, 1), key='-metrics-'),
             sg.Button('NetMake', key='-NetMake-')],
        ]

        FlowChartPrint = [
            [sg.Button('FlowChart', key='-FlowChartPrint-')],
        ]

        NuralNet = [
            [sg.Text('学習用データ')],
            [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgLRN-')],
            [sg.Text('学習用ラベル')],
            [sg.Input(size=(30, 1)), sg.FileBrowse(key='-orgTrg-')],
            [sg.Text('Validation用データ')],
            [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valRLN-')],
            [sg.Text('Validation用ラベル')],
            [sg.Input(size=(30, 1)), sg.FileBrowse(key='-valTrg-')],
            [sg.Radio('標準化', 'feature', size=(10, 1), key='-feature_0-'),
             sg.Radio('正規化', 'feature', size=(10, 1), key='-feature_1-'),
             sg.Radio('両方', 'feature', size=(10, 1),  key='-feature_2-')],
            [sg.Text('Batch Size'), sg.Text('epochs')],
            [sg.InputText('128', size=(10, 1), key='-Batch-'),
             sg.InputText('100', size=(10, 1), key='-epochs-'),
             sg.Button('Training', key='-TrainingRUN-')],
        ]

        NetEvaluate = [
            [sg.Text('テスト用データ')],
            [sg.Input(size=(30, 1)), sg.FileBrowse(key='-tstRLN-')],
            [sg.Text('テスト用ラベル')],
            [sg.Input(size=(30, 1)),
             sg.FileBrowse(key='-tstTrg-'),
             sg.Button('Test', key='-TestRUN-')],
        ]

        layout = [
            [sg.Frame('NetMake', NetMake),
             sg.Frame('NuralNet', NuralNet),
             sg.Column(FlowChartPrint)],
            [sg.Frame('DataConv', dataConv)],
            [sg.Quit()],
        ]

        InLayer = None # 層が追加されているか判定するための変数
        window = sg.Window('NeuralNet', layout, default_element_size=(40, 1))
        while True:
            event, values = window.Read(timeout=10)
            if event is None or event == 'Quit':
                break
            #---------------------------------
            # DataMake_click
            #---------------------------------
            if event is '-DataMake-':
                self.DataMake_click(importDirPath=values['-importDirPath-'],
                               outputDirPath=values['-outputDirPath-'],
                               org_FileName=values['-orgDataFileName-'],
                               lrn_FileName=values['-lrnDataFileName-'],
                               tst_FileName=values['-tstDataFileName-'],
                               digit=values['-Digit-'])
            #---------------------------------
            # Networkの層を追加する。
            #---------------------------------
            elif event is '-LayerAdd-':
                InLayer = self.LayerAdd_click(layerName=values['-LayerName-'],
                                              Node=values['-Node-'],
                                              weight=values['-weightInit-'],
                                              bias=values['-biasInit-'],
                                              activation=values['-activation-'])
            elif event is '-NetMake-':
                if InLayer is None or InLayer in 'input':
                    print('層もしくは隠れ層がセットされていません。')
                    continue
                self.NetMake_click(values['-loss-'],
                                   values['-optimizer-'], 
                                   values['-metrics-'])
                #NetMake_click(values['-loss-'])
            #----------------------------
            # Trainig_click
            #----------------------------
            elif event is '-TrainingRUN-':
                if InLayer is None or InLayer in 'input':
                    print('層もしくは隠れ層がセットされていません。')
                    continue
                #----------------------------
                # ラジオボタンの条件分岐
                #----------------------------
                if values['-feature_0-'] is True:
                    feature = 0
                elif values['-feature_1-'] is True:
                    feature = 1
                elif values['-feature_2-'] is True:
                    feature = 2
                else:
                    feature = None
                #----------------------------
                # Validationの条件分岐
                #----------------------------
                if (values['-valRLN-'] != '' and values['-valTrg-'] != ''):
                    val = (values['-valRLN-'], values['-valTrg-'])
                else:
                    val = None

                self.Training_click((values['-orgLRN-'], values['-orgTrg-']),
                               values['-Batch-'], values['-epochs-'], feature, val)

            elif event is '-FlowChartPrint-':
                    self.FlowChartPrint_click()

#def NetWorkTree():

if __name__ == "__main__":
    NuralNet_APP()
