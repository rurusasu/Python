from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r'D:\myfile\My_programing\python\画像認識\4032x3024_IMG_0664.JPG').convert('RGB')
img = np.array(img)
print(img.shape)
#print(img)

plt.imshow(img)
plt.show()

