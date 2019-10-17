# cording : utf-8

import tensorflow as tf
import tensorflow.contrib.eager as tfe


tfe.enable_eager_execution()


print("\n変数\n=========================")

w = tfe.Variable(tf.zeros([3, 2])) # 変数（行列）wの宣言. 初期値はゼロ行列

tf.global_variables_initializer()

print("w = %s" % w)
tf.assign(w, tf.ones([3, 2]))

wn = w
print("w = %s" % wn)

# 変数
# =========================
# w = <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
# array([[0., 0.],
#        [0., 0.],
#        [0., 0.]], dtype=float32)>
# w = <tf.Variable 'Variable:0' shape=(3, 2) dtype=float32, numpy=
# array([[1., 1.],
#       [1., 1.],
#       [1., 1.]], dtype=float32)>


# 値をnumpy型で取り出したいとき
print(wn.numpy())