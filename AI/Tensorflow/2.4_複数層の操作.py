import numpy as np
import tensorflow as tf
import os

sess = tf.Session()

# サンプルの2D画像を作成　shape=[画像の数, 高さ, 幅, チャネル]
x_shape = [1, 4, 4, 1]
x_val = np.random.uniform(size=x_shape) 
#print(x_val)

#プレースホルダを作成
x_data = tf.placeholder(tf.float32, shape=x_shape)

# 移動平均ウインドウを設定
my_filter = tf.constant(0.25, shape=[2, 2, 1, 1])
my_strides = [1, 2, 2, 1]
mov_avg_layer = tf.nn.conv2d(x_data, my_filter, my_strides, padding='SAME', name='Moving_AVG_Window')

# 移動平均ウインドウの2x2出力を操作するカスタム層を定義
def custom_layer(input_matrix):
    input_matrix_sqeezed =tf.squeeze(input_matrix)
    A = tf.constant([[1., 2.], [-1., 3.]])
    b = tf.constant(1., shape=[2, 2])
    temp1 = tf.matmul(A, input_matrix_sqeezed) # 内積の計算（ブロードキャストなし）
    temp = tf.add(temp1, b) # Ax + b
    return(tf.sigmoid(temp))

# 新しい層を計算グラフに追加
with tf.name_scope('Custom_Layer') as scope:
    custom_layer1 = custom_layer(mov_avg_layer)

print(sess.run(custom_layer1, feed_dict={x_data: x_val}))

