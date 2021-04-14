#!/usr/bin/env python

import DobotDllType as dType

"""-------The DoBot Control Class-------
Variables:
suction = Suction is currently on/off
picking: shows if the dobot is currently picking or dropping an item
api = variable for accessing the dobot .dll functions
home% = home position for %
                                  """

CON_STR = {
    dType.DobotConnect.DobotConnect_NoError:  "DobotConnect_NoError",
    dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
    dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied"
}


#Main control class for the DoBot Magician.
class Dobot:
    def __init__(self, homeX, homeY, homeZ):
        self.suction = False
        self.picking = False
        dType.dSleep(50)
        self.api = dType.load()
        dType.dSleep(50)
        self.homeX = homeX
        self.homeY = homeY
        self.homeZ = homeZ
        self.connected = False
        dType.dSleep(50)
        self.dobotConnect()
        dType.dSleep(50)

    def __del__(self):
        self.dobotDisconnect()

    #Attempts to connect to the dobot
    def dobotConnect(self):
        if(self.connected):
            print("You're already connected")
        else:
            state = dType.ConnectDobot(self.api, "", 115200)[0]
            if(state == dType.DobotConnect.DobotConnect_NoError):
                print("Connect status:",CON_STR[state])
                dType.SetQueuedCmdClear(self.api)
                dType.SetHOMEParams(self.api, self.homeX, self.homeY, self.homeZ, 0, isQueued = 1)
                dType.SetPTPJointParams(self.api, 200, 200, 200, 200, 200, 200, 200, 200, isQueued = 1)
                dType.SetPTPCommonParams(self.api, 100, 100, isQueued = 1)
                dType.SetHOMECmd(self.api, temp = 0, isQueued = 1)
                self.connected = True
                return self.connected
            else:
                print("Unable to connect")
                print("Connect status:",CON_STR[state])
                return self.connected

    #Returns to home location and then disconnects
    def dobotDisconnect(self):
        self.moveHome()
        dType.DisconnectDobot(self.api)

    #Delays commands
    def commandDelay(self, lastIndex):
        dType.SetQueuedCmdStartExec(self.api)
        dType.dSleep(100)
        while lastIndex > dType.GetQueuedCmdCurrentIndex(self.api)[0]:
            position1 = dType.GetPose(self.api)
            dType.dSleep(100)
            position2 = dType.GetPose(self.api)
            if(position1 == position2):
                print(position1,position2)
                break
        dType.SetQueuedCmdStopExec(self.api)

    #Toggles suction peripheral on/off
    def toggleSuction(self):
        lastIndex = 0
        if(self.suction):
            lastIndex = dType.SetEndEffectorSuctionCup( self.api, True, False, isQueued = 0)[0]
            self.suction = False
        else:
            lastIndex = dType.SetEndEffectorSuctionCup(self.api, True, True, isQueued = 0)[0]
            self.suction = True
        self.commandDelay(lastIndex)

    def move(self,x,y,z,a):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode,x, y, z, a, isQueued = 1)[0]
        self.commandDelay(lastIndex)
        
    def getPose(self):
        class Pose:
            pass
        pose = Pose()
        pose.arr = dType.GetPose(self.api)
        pose.x = pose.arr[0]
        pose.y = pose.arr[1]
        pose.z = pose.arr[2]
        return pose
        
    #Moves arm to X/Y/Z Location
    def moveArmXY(self,x,y):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, x, y, self.homeZ, 0,isQueued=0)[0]
        self.commandDelay(lastIndex)

    #Returns to home location
    def moveHome(self):
        lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, self.homeX, self.homeY, self.homeZ, 0)[0]
        self.commandDelay(lastIndex)

    #Toggles between hover and item level
    def pickToggle(self, itemHeight):
        lastIndex = 0
        positions = dType.GetPose(self.api)
        if(self.picking):
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], self.homeZ, 0)[0]
            self.picking = False
        else:
            lastIndex = dType.SetPTPCmd(self.api, dType.PTPMode.PTPMOVLXYZMode, positions[0], positions[1], itemHeight, 0)[0]
            self.picking = True
        self.commandDelay(lastIndex)
