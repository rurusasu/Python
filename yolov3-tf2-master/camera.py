# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 09:15:36 2021

@author: PC-2017-4
"""

import cv2

#frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_LINEAR)
#frame = np.array(frame, dtype='float32')

class Camera:
    def __init__(self, num=0):
        self.capture = cv2.VideoCapture(num)
        if self.capture.isOpened() is False:
            raise IOError

    def __del__(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def getImage(self):
        ret, frame = self.capture.read()
        if ret is False:
            raise IOError
        return frame
     
    def showImage(self):
        while(True):
            try:
                ret, frame = self.capture.read()
               #cv2.imshow('frame',frame)
                cv2.waitKey(1) 
            except KeyboardInterrupt:
                # 終わるときは CTRL + C を押す
                break
        self.capture.release()
        cv2.destroyAllWindows()
        

            
   


