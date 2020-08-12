# cording: utf-8

import sys
import os
sys.path.append(os.getcwd())

import DobotDllType as dType
from common.DobotFunction import initDobot, _OneAction, csv_write
from ctypes import *

class Dobot:
    def __init__(self):
        # Load Dll
        self. api = cdll.LoadLibrary('DobotDll.dll')
        
        self.CON_STR = {
            dType.DobotConnect.DobotConnect_NoError:  'DobotConnect_NoError',
            dType.DobotConnect.DobotConnect_NotFound: 'DobotConnect_NotFound',
            dType.DobotConnect.DobotConnect_Occupied: 'DobotConnect_Occupied'
        }

    def connect(self):
        # Dobot Connect
        # ConectDobot(const char* pointName, int baudrate)
        state = dType.ConnectDobot(self.api, "", 115200)[0]
        if (state != self.CON_STR[dType.DobotConnect.DobotConnect_NoError]):
            print('Dobot Connect', 'Dobotに接続できませんでした。')
            return

        #Clean Command Queued
        dType.SetQueuedCmdClear(self.api)

        #Async Motion Params Setting
        dType.SetHOMEParams(self.api, 150, -200, 100, 0, isQueued=1)

        #Async Home
        dType.SetHOMECmd(self.api, temp=0, isQueued=1)

        initDobot(self.api)

    
    #-----------------------------------
    # Dobotの動作用_汎用関数
    #-----------------------------------
    def Operation(self, file_name, axis, volume=1, initPOS=None):
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
            pose = dType.GetPose(self.api)

        if (axis in axis_list):
            if (axis == 'x'):
                _OneAction(self.api, dType.PTPMode.PTPMOVLXYZMode,
                        pose[0] + volume, pose[1], pose[2], pose[3])
            elif (axis == 'y'):
                _OneAction(self.api, dType.PTPMode.PTPMOVLXYZMode,
                        pose[0], pose[1] + volume, pose[2], pose[3])
            elif (axis == 'z'):
                _OneAction(self.api, dType.PTPMode.PTPMOVLXYZMode,
                        pose[0], pose[1], pose[2] + volume, pose[3])
            else:
                print('rは実装されていません。')
        else:
            print('移動軸に問題があります！')

        # 座標をファイルへ書き込む
        csv_write(file_name, dType.GetPose(self.api))


    def getpose(self):
        return dType.GetPose(self.api)
