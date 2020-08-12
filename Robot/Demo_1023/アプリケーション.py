# cording: utf-8

import threading
import DobotDllType as dType
from ctypes import *

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}


# Load Dll
api = cdll.LoadLibrary("DobotDll.dll")

# connect Dobot
state = dType.ConnectDobot(api, "", 115200)[0] # ConectDobot(const char* pointName, int baudrate)
print("Connect status:", CON_STR[state])

if (state == dType.DobotConnect.DobotConnect_NoError):

    # Clean Command Queued
    dType.SetQueuedCmdClear(api)

    # Async Motion Params Setting
    """
    SetHomeParams(
    api,
    x,
    y,
    z,
    r,
    isQueued = 0
    )
    """
    dType.SetHOMEParams(api, 250, 0, 50, 0, isQueued = 1)
    
    """
    SetPTPJointParams(
    api, 
    j1Velocity,
    j2Velocity,    
    j3Velocity,    
    j4Velocity,
    j1Acceleration,
    j2Acceleration,
    j3Acceleration,
    j4Acceleration,
    isQueued = 0
    )
    """
    dType.SetPTPJointParams(api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1) 

    """
    SetPTPCommonParams(
    api, 
    velocityRatio,
    accelerationRatio,
    isQueued = 0
    )
    """
    dType.SetPTPCommonParams(api, 100, 100, isQueued = 1)

    # Async Home
    dType.SetHOMECmd(api, temp = 0, isQueued = 1)

    # Async PTP Motion
    """
    ここに人工知能での処理を記述。


    """
    lastIndex = dType.SetPTPCmd(api, dType.PTPMode.PTPMOVLXYZMODE, )