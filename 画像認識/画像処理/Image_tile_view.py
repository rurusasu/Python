import os
import glob
import cv2
import numpy as np

im_list = []
f_path = os.getcwd() + '\\*.jpg'
f_list = glob.glob(f_path)
print(f_list)


for f in f_list:
    src_list
    for i in range(3):
        src = cv2.imread(f)
        im_list.append(src)