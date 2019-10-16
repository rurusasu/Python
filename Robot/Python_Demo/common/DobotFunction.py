# cording: ustf-8

import csv
import DobotDllType as dType


#-----------------
# Dobotの初期化
#-----------------
def initDobot():
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
def Operation(api, axis, volume=1):
    """
    A function that sends a motion command in any direction

    Parameters
    ----------
    api : CDLL
    axis : str
        移動方向
    volume : int
        移動量
    """
    axis_list = ['x', 'y', 'z', 'r']
    pose = dType.GetPose(api)

    if (axis in axis_list):
        if (axis == 'x'):
            _OneAction(api, dType.PTPMode.PTPMOVLXYZMode, pose[0] + volume, pose[1], pose[2], pose[3])
        elif (axis == 'y'):
            _OneAction(api, dType.PTPMode.PTPMOVLXYZMode, pose[0], pose[1] + volume, pose[2], pose[3])
        elif (axis == 'z'):
            _OneAction(api, dType.PTPMode.PTPMOVLXYZMode, pose[0], pose[1], pose[2] + volume, pose[3])
        else:
            print('rは実装されていません。')
    else:
        print('移動軸に問題があります！')


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


#-----------------------------------
# Dobotの動作用関数
#-----------------------------------
def roop_plusX(api, x, y, z, r, roop):
    counter = x
    list = []

    #Async PTP Motion
    for j in range(1, roop + 1):
        _OneAction(api, dType.PTPMode.PTPMOVLXYZMode, counter + j, y, z, r)
        list.append(GetPose(api))

    csv_write(pose)
    counter += j
    return counter


def roop_minusX(api, x, y, z, r, roop):
    counter = x
    list = []

    #Async PTP Motion
    for j in range(1, roop + 1):
        _OneAction(api, dType.PTPMode.PTPMOVLXYZMode, counter - j, y, z, r)
        list.append(GetPose(api))

    csv_write(pose)
    counter -= j
    return counter


def act_plusY(api, x, y, z, r, roop = 1):
    counter = y

    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode,
                                    x, counter + 1, z, r, isQueued=1)[0]
    _act(api, lastIndex)
    pose = dType.GetPose(api)
    
    file_write(pose)

    counter += 1
    return counter


def act_minusZ(api, x, y, z, r):
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode,
                                x, y, z, r, isQueued=1)[0]
    _act(api, lastIndex)


#----------------------------------
# CsvFileへの書き込み関数
#----------------------------------
def csv_write(filename, data):
        """Write Data to csv file"""
        if (data == None):  # 書き込むデータが無いとき
            return
        with open(filename, 'a', encoding='utf_8', errors='', newline='') as f:
            # ファイルへの書き込みを行う
            if(_wirte(f, data) == None):
                print('書き込みが完了しました。')


def _wirte(f, data):
    """write content"""

    error = 1  # エラーチェック用変数
    witer = csv.writer(f, lineterminator='\n')
    error = witer.writerows(data)

    return error  # エラーが無ければNoneを返す
