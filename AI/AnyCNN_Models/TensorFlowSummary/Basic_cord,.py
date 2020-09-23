import tensorflow as tf

import sys
import os

sys.path.append(".")

logdir_name = "log"

logdir = os.path.dirname(os.path.abspath(__file__))
logdir = logdir + os.sep + logdir_name
print(logdir)

writer = tf.summary.create_file_writer(logdir=logdir)
writer.set_as_default()

x = tf.Variable(1, name="x")

for i in range(100):
    x.assign_add(1)
    tf.summary.scalar("x", x, step=i, description="first valiable")

