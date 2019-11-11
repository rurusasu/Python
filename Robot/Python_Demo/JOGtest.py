import threading
import DobotDllType as dType
from DobotDll import*
from ctypes import*

#import os, sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"}

#Load Dll
api = dType.load()

#Connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
print("Connect status:",CON_STR[state])

if (state == dType.DobotConnect.DobotConnect_NoError): #stateがDobotConnect_NoErrorに等しい時

    #Clean Command Queued
    dType.SetQueuedCmdClear(api)

    #Async Motion Params Setting
    
    # 初期の位置決め
    dType.SetHOMEParams(api, 200, 0, 100, 0, isQueued = 1)
    """
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)
    """
    
    dType.SetJOGJointParams(api, 200, 200, 200, 200,
                            200, 200, 200, 200, isQueued=1)      # 関節座標系での各モータの速度および加速度の設定
    #dType.SetJOGCoordinateParams(api, 200, 200, 200, 200,
                                 #200, 200, 200, 200, isQueued=1)  # デカルト座標系での各方向への速度および加速度の設定
    # JOG動作の速度、加速度の比率を設定
    dType.SetJOGCommonParams(api, 100, 100, isQueued=1)

    #Async Home
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    """
    #Async PTP Motion
    for i in range(0, 5):
        if i % 2 == 0:
            offset = 50
        else:
            offset = -50
        lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMode, 200 + offset, offset, offset, offset, isQueued = 1)[0]
    """
    # Async JOG Motion
    #for i in range(1, 4):
        #lastIndex = dType.SetJOGCmd(api, 1, 1, isQueued = 1)
    joint = c_byte(1)
    angle = c_byte(7)
    lastIndex = dType.SetJOGCmd(api, joint, angle, isQueued = 1)
    #lastIndex = dType.SetJOGCmd(api, 1, 8, isQueued = 1)
    print('動作を開始します。')
    #  This  EMotor
    #  This  EMotor
    #  This  EMotor
    #  This  EMotor

    # dType.SetEMotor(api, 0, 1, 10000, isQueued=1)
    # dType.SetEMotorS(api, 0, 1, 10000, 20000,isQueued=1)



    #Start to Execute Command Queued
    dType.SetQueuedCmdStartExec(api)

    #Wait for Executing Last Command
    while lastIndex > dType.GetQueuedCmdCurrentIndex(api)[0]:
        dType.dSleep(100)

    #Stop to Execute Command Queued
    dType.SetQueuedCmdStopExec(api)

#Disconnect Dobot
dType.DisconnectDobot(api)
