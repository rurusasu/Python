from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r'E:\Data\H31_Miki\python\Data\pen\0\IMG_E0663.JPG').convert('RGB')
img = np.array(img)
print(img.shape)
#print(img)

plt.imshow(img)
plt.show()

