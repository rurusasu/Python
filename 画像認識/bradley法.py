import cv2
import numpy as np

def Bradley_threshold(src):
    new_img = src.copy()
    (height, width, channel) = new_img.shape
    S = int(width / 8)
    s2 = int(S/2)
    t = 0.15
    sum = 0
    count = 0
    res = integral_image = np.zeros((height, width))

    if(channel >= 3):
        new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)

    # 積分画像作成
    for i in range(height):
        sum = 0
        for j in range(width):
            sum += new_img[i, j]
            if (i == 0):
                integral_image[i, j] = sum
            else:
                integral_image[i, j] = integral_image[i-1, j] + sum

    print(integral_image[479, 639])
    print(np.sum(src))
    
    for i in range(height):
        for j in range(width):
            x1 = j - s2
            x2 = j + s2
            y1 = i - s2
            y2 = i + s2

            if (x1 < 0): x1 = 0
            if (x2 >= width): x2 = width-1
            if (y1 < 0): y1 = 0
            if (y2 >= height): y2 = height-1

            count = (x2-x1) * (y2-y1)

            sum = integral_image[x2, y2] - integral_image[x2, y1-1] - \
                integral_image[x1-1, y2] + integral_image[x1-1, y1-1]

            res[i, j] = 0




def save_frame_camera_key(device_num, basename, ext='jpg', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    while True:
        ret, frame = cap.read()
        frame = Bradley_threshold(frame)
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyWindow(window_name)


save_frame_camera_key(0, 'camera_capture')


