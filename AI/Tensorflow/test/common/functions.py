import numpy as np
import tensorflow as tf

def Relu(x):
    return (tf.nn.relu(x, name='relu'))