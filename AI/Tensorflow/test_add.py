# cording : utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe


tfe.enable_eager_execution()


const1 = tf.constant(2)
const2 = tf.constant(3)
add_op = tf.add(const1, const2) # 2 + 3
print(add_op)
# tf.Tensor(5, shape=(), dtype=int32)


print(type(add_op))
# tensorflow.python.framework.ops.EagerTensor

add_op2 = add_op + 3
print(add_op2)
# tf.Tensor(8, shape=(), dtype=int32)

print(type(add_op2))
# tensorflow.python.framework.ops.EagerTensor

add_op3 = add_op2.numpy()
print(add_op3)
# 8

print(type(add_op3))
# numpy.int32