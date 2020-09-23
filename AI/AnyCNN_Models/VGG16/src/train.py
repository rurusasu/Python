import os
import sys

sys.path.append(".")
sys.path.append("..")

import tensorflow as tf

from lib.config import cfg
from models.vgg16 import load_model
from lib.utils import load_dataset, get_args, get_current_time, gpu_setup


def builtin_train(args):
    # load dataset and model
    (train_images, train_labels), (test_images, test_labels) = load_dataset(args.data)
    input_shape = train_images[: args.batch_size, :, :, :].shape
    output_size = max(train_labels) + 1
    model = load_model(input_shape=input_shape, output_size=output_size)

    # loss, optimizer, metrics, setting
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # model.summary()

    # set tensorboard configs
    # logdir = os.path.join(args.logdir, get_current_time())
    logdir_name = "log"

    logdir = os.path.dirname(os.path.abspath(__file__))
    logdir = logdir + os.sep + logdir_name

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

    # dataset config (and validation, callback config)
    fit_params = {}
    fit_params["batch_size"] = args.batch_size
    fit_params["epochs"] = args.max_epoch
    if args.steps_per_epoch:
        fit_params["steps_per_epoch"] = args.steps_per_epoch
    fit_params["verbose"] = 1
    fit_params["callbacks"] = [tensorboard_callback]
    fit_params["validation_data"] = (test_images, test_labels)

    # start train and test
    gpu_setup()
    model.fit(train_images, train_labels, **fit_params)


if __name__ == "__main__":
    args = get_args()
    args.batch_size = 100
    args.max_epoch = 5

    builtin_train(args)
