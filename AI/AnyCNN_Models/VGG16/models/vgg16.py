import os
import tensorflow as tf
from tensorflow.keras import layers, Model

# from src.utils import load_dataset, get_args, get_current_time


class CBR(Model):
    def __init__(self, filters, kernel_size, strides):
        super().__init__()

        params = {
            "filters": filters,
            "kernel_size": kernel_size,
            "strides": strides,
            "padding": "same",
            "use_bias": True,
            "kernel_initializer": "he_normal",
        }

        self.layers_ = [
            layers.Conv2D(**params),
            layers.BatchNormalization(),
            layers.ReLU(),
        ]

    def call(self, inputs):
        for layer in self.layers_:
            inputs = layer(inputs)
        return inputs


class VGG16(Model):
    def __init__(self, output_size=1000):
        super().__init__()
        self.laeyrs_ = [
            CBR(64, 3, 1),
            CBR(64, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(128, 3, 1),
            CBR(128, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            CBR(256, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            CBR(512, 3, 1),
            layers.MaxPool2D(2, padding="same"),
            layers.Flatten(),
            layers.Dense(4096),
            layers.Dense(4096),
            layers.Dense(output_size, activation="softmax"),
        ]

    def call(self, inputs):
        for layer in self.laeyrs_:
            inputs = layer(inputs)
        return inputs


def load_model(input_shape=None, output_size=10):
    """モデルを読み込む
    Args:
        output_size(int):
    Returns:
        model(Object): tf.keras.Model を継承したオブジェクト
    """

    model = VGG16(output_size)
    return model
