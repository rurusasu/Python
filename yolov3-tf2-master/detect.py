import tensorflow as tf
from tensorflow.compat.v1 import flags
#import time

import cv2
#import numpy as np

from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs



class Yolo():
    def __init__(self):
        try:
            flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
            flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                                'path to weights file')
            flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
            flags.DEFINE_integer('size', 416, 'resize images to')
            flags.DEFINE_string('image', './data/girl.png', 'path to input image')
            flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
            flags.DEFINE_string('output', './output.jpg', 'path to output image')
            flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
        except NameError:
            print("再代入禁止")
        self.yolo = YoloV3(classes=flags.FLAGS.num_classes)
        self.yolo.load_weights(flags.FLAGS.weights).expect_partial()
        #logging.info('weights loaded')
    
        self.class_names = [c.strip() for c in open(flags.FLAGS.classes).readlines()]
        #logging.info('classes loaded')


    def detect(self, img_raw):
        img_tf = tf.Variable(img_raw)
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, flags.FLAGS.size)
        
        boxes, scores, classes, nums = self.yolo(img)

        img = cv2.cvtColor(img_tf.numpy(),cv2.COLOR_BGR2GRAY)
        img = draw_outputs(img, (boxes, scores, classes, nums), self.class_names)
        #cv2.imwrite(flags.FLAGS.output, img)
        #print(boxes)


        return img, boxes, nums
