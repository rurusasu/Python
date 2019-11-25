from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

img = Image.open(r'D:\myfile\My_programing\python\画像認識\4032x3024_IMG_0664.JPG').convert('RGB')


def grayscale(IMG_Data):
    """
    グレースケールに変換する関数

    Parameters
    ----------
    IMG_Data_list : list
        画像データのリスト

    Retern
    ------
    img : list
        変換後の画像のリスト
    """
    if IMG_Data.mode != 'RGB':
        IMG_Data = IMG_Data.convert('RGB') # any format to RGB
    rgb = np.array(IMG_Data, dtype='float32')

    rgbL = pow(rgb/255.0, 2.2)
    r, g, b = rgbL[:,:,0], rgbL[:,:,1], rgbL[:,:,2]
    grayL = 0.299*r + 0.587*g + 0.114*b #BT.601
    gray = pow(grayL, 1.0/2.2)*255
    
    return Image.fromarray(gray.astype('uint8'))

img = grayscale(img)

img = np.array(img)
plt.imshow(img)
plt.show()
