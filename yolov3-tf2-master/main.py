# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 14:11:50 2021

@author: PC-2017-4
"""

import detect
import camera
import dobot
import cv2
import tensorflow as tf


def main():
    yolo = detect.Yolo()
    cm = camera.Camera(num=1)
    homex, homey, homez = 200, 0, 70
    dbt = dobot.Dobot(homex, homey, homez)

    while True:
        try:
            img = cm.getImage()
            detectedImg, boxes, nums = yolo.detect(img)
            gain = 50

            if(nums[0]>0):
                y = (boxes[0][0][0] + boxes[0][0][2] - 1) * gain
                x = (boxes[0][0][1] + boxes[0][0][3] - 1) * gain
                print("x,y:",x,y)
                pose = dbt.getPose()
                dbt.move(int(pose.x - x), int(pose.y - y), homez,0)
                print("dobot x,y:pose.x,pose.y", pose.x, pose.y)

            cv2.imshow('frame',detectedImg)
            cv2.waitKey(1)
        except KeyboardInterrupt:
            dbt.dobotDisconnect()
            break

if __name__ == "__main__":
    main()