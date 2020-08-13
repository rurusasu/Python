import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf

# GPUの使用を制限
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for k in range(len(physical_devices)):
        tf.config.experimental.set_memory_growth(physical_devices[k], True)
        print('memory growth:', tf.config.experimental.get_memory_growth(
            physical_devices[k]))
else:
    print("Not enough GPU hardware devices available")

# 入力画像を変換
img_path = '000001.jpg'
img = image.load_img(img_path, target_size=(224, 224))
# plt.imshow(img)
# plt.show()
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = preprocess_input(img)

print('IMAGE: %s' % str(img.shape))  # IMAGE: (1, 244, 244, 3)

# vgg16をロード
model = VGG16(weights='imagenet', include_top=False)
layers = model.layers[1:19]
layer_outputs = [layer.output for layer in layers]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activation_model.summary()

activations = activation_model.predict(img)
for i, activation in enumerate(activations):
    print("%2d: %s" % (i, str(activation.shape)))
#features = model.predict(x)

# print('推定が終了しました。')
# print(features)
