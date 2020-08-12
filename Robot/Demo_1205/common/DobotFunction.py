# cording: ustf-8

import sys, os
sys.path.append(os.getcwd())
import csv
import DobotDllType as dType


#-----------------
# Dobotの初期化
#-----------------
def connect(api, CON_STR):
        # Dobot Connect
        # ConectDobot(const char* pointName, int baudrate)
        state = dType.ConnectDobot(api, "", 115200)[0]
        if (state != CON_STR[dType.DobotConnect.DobotConnect_NoError]):
            print('Dobot Connect', 'Dobotに接続できませんでした。')
            return

        #Clean Command Queued
        dType.SetQueuedCmdClear(api)

        #Async Motion Params Setting
        dType.SetHOMEParams(api, 150, -200, 100, 0, isQueued=1)

        #Async Home
        dType.SetHOMECmd(api, temp=0, isQueued=1)

        initDobot(api)



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
# 直交座標系での動作
def Operation(api, file_name, axis, volume=1, initPOS=None):
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
        if (initPOS != None):
            pose = initPOS
        else:
            pose = dType.GetPose(api)

        if (axis in axis_list):
            if (axis == 'x'):
                _OneAction(api, pose[0] + volume, pose[1], pose[2], pose[3])
            elif (axis == 'y'):
                _OneAction(api, pose[0], pose[1] + volume, pose[2], pose[3])
            elif (axis == 'z'):
                _OneAction(api, pose[0], pose[1], pose[2] + volume, pose[3])
            else:
                print('rは実装されていません。')
        else:
            print('移動軸に問題があります！')

        # 座標をファイルへ書き込む
        csv_write(file_name, dType.GetPose(api))


# 1回動作指令を出す関数
def OneAction(api, x=None, y=None, z=None, r=None, mode=dType.PTPMode.PTPMOVLXYZMode):
        """One step operation"""
        if (x is None or y is None or z is None or r is None):
            pose = dType.GetPose(api)
            if x is None:
                x=pose[0]
            if y is None:
                y=pose[1]
            if z is None:
                z=pose[2]
            if r is None:
                r=pose[3]
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
