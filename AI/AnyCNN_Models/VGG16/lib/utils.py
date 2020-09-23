import sys

sys.path.append(".")
sys.path.append("..")

import tensorflow as tf
import argparse

from datetime import datetime


def gpu_setup(gpu_num=0):
    """使用する GPU を設定する
    Args:
        gpu_num(int):
    """
    physical_devices = tf.config.list_physical_devices("GPU")

    if physical_devices:
        try:
            print("# Found {} GPU(s)".format(len(physical_devices)))

            for device in physical_devices:
                tf.config.set_visible_devices(physical_devices[gpu_num], "GPU")
                tf.config.experimental.set_memory_growth(
                    physical_devices[gpu_num], True
                )
                print(
                    "# {} memory growth: {}".format(
                        device, tf.config.experimental.get_memory_growth(device)
                    )
                )

        except RuntimeError as e:
            print(e)

    else:
        print("Not enough GPU hardware devices available")


def get_current_time():
    """現在時刻を取得します
    Returns:
        (str):
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_dataset(name):
    """データセットを読み込む。
    Args:
        name(str):
    Returns:
        (tuple): (np.ndarray, np.ndarray)
        (tuple): (np.ndarray, np.ndarray)
    """

    data = eval(f"tf.keras.datasets.{name.lower().replace('-', '_')}")
    (train_images, train_labels), (test_images, test_labels) = data.load_data()

    train_images = train_images.astype("float32") / 255
    test_images = test_images.astype("float32") / 255

    if train_images.ndim == 3:
        train_images = train_images[:, :, :, tf.newaxis]
        test_images = test_images[:, :, :, tf.newaxis]

    train_labels = train_labels.astype("float32")
    test_labels = test_labels.astype("float32")

    return (train_images, train_labels), (test_images, test_labels)


def get_args():
    """引数をパースして取得します。
    Returns:
        args(argparse.Namespace)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="mnist", choices=["mnist"])
    parser.add_argument("--logdir", "-l", default="./logs")
    parser.add_argument("--batch-size", "-b", type=int, default=10)
    parser.add_argument("--max-epoch", "-e", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    parser.add_argument("--custom-train", action="store_true")
    parser.add_argument("--augmentation", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    gpu_setup()
