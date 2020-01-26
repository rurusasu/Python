from glob import glob
import numpy as np
import cv2, PySimpleGUI as sg
import argparse
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num_classes = 2
img_height, img_width = 32, 32
tf.set_random_seed(0)


layout = [[sg.Text]]