'''
给 img.jpg 图像添加椒盐噪声
'''
import cv2
import random
import numpy as np


def pepper_salt(src, percetage):
    noiseimg = src
    noisenum = int(percetage*src.shape[0]*src.shape[1])
    for i in range(noisenum):
        randx = np.random.random_integers(0,src.shape[0]-1)
        randy = np.random.random_integers(0,src.shape[1]-1)
        if np.random.random_integers(0,1)<=0.5:
            noiseimg[randx,randy] = 0
        else:
            noiseimg[randx,randy]=255
    return noiseimg


def main():
    img = cv2.imread('./img.jpg', 0)
    img_new = pepper_salt(img, 0.3)  # 设置椒盐噪声比例, 可设置为0.1, 0.2, 0.3
    cv2.imwrite('./img_salt.jpg', img_new)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
