import cv2
import sys, os
sys.path.append(os.getcwd())


def main():
    org_img = cv2.imread('resource/lena.png')
    r, g, b = cv2.split(org_img) # r,g,bの画素値をそれぞれ抽出する
    cv2.imshow(r)

if __name__ == "__main__":
    main()
