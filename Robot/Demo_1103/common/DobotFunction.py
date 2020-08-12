# cording: ustf-8

import sys, os
sys.path.append(os.getcwd())
import csv
import DobotDllType as dType


#-----------------
# Dobotの初期化
#-----------------
def initDobot(api):
    # Clean Command Queued
    dType.SetQueuedCmdClear(api)

    # デバイスのシリアルナンバーを取得する
    dSN = dType.GetDeviceSN(api)
    print(dSN)

    # デバイス名を取得する
    dName = dType.GetDeviceName(api)
    print(dName)

    # デバイスのバージョンを取得する
    majorV, minorV, revision = dType.GetDeviceVersion(api)
    print(majorV, minorV, revision)

    # JOGパラメータの設定
    dType.SetJOGJointParams(api, 200, 200, 200, 200,
                            200, 200, 200, 200, isQueued=1)      # 関節座標系での各モータの速度および加速度の設定
    dType.SetJOGCoordinateParams(api, 200, 200, 200, 200,
                                 200, 200, 200, 200, isQueued=1)  # デカルト座標系での各方向への速度および加速度の設定
    # JOG動作の速度、加速度の比率を設定
    dType.SetJOGCommonParams(api, 100, 100, isQueued=1)

    # PTPパラメータの設定
    dType.SetPTPJointParams(api, 200, 200, 200, 200,
                            200, 200, 200, 200, isQueued=1)           # 関節座標系の各モータの速度および加速度を設定
    # デカルト座標系での各方向への速度および加速度の設定
    dType.SetPTPCoordinateParams(api, 200, 200, 200, 200, isQueued=1)
    # PTP動作の速度、加速度の比率を設定
    dType.SetPTPCommonParams(api, 100, 100, isQueued=1)


#-----------------------------------
# Dobotの動作用_汎用関数
#-----------------------------------
# 1回動作指令を出す関数
def _OneAction(api, mode, x, y, z, r):
    """One step operation"""
    lastIndex = dType.SetPTPCmd(api, mode,
                                x, y, z, r, isQueued=1)[0]
    _Act(api, lastIndex)


def _Act(api, lastIndex):
    """Function to execute command"""
    #キューに入っているコマンドを実行
    dType.SetQueuedCmdStartExec(api)

    #Wait for Executing Last Command
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)

    #キューに入っているコマンドを停止
    dType.SetQueuedCmdStopExec(api)


#----------------------------------
# CsvFileへの書き込み関数
#----------------------------------
def csv_write(filename, data):
        """Write Data to csv file"""
        if (data == None):  # 書き込むデータが無いとき
            return
        array = [str(row) for row in data]
        with open(filename, 'a', encoding='utf_8', errors='', newline='') as f:
            # ファイルへの書き込みを行う
            if(_wirte(f, array) == None):
                print('x=%f,  y=%f,  z=%f,  r=%f' %(data[0], data[1], data[2], data[3]))
                #print('書き込みが完了しました。')
            else:
                print('ファイルの書き込みに失敗しました。')


def _wirte(f, data):
    """write content"""
    error = 1  # エラーチェック用変数
    witer = csv.writer(f, lineterminator='\n')
    error = witer.writerows([data])

    return error  # エラーが無ければNoneを返す
