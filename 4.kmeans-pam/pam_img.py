'''
利用PAM算法实现图像分割
'''
import numpy as np
import random
import cv2


def func_loss(present_center, pre_center):
    return np.sum((np.array(present_center) - np.array(pre_center))**2)


def classifier(input_signal, center):
    input_row, input_col = input_signal.shape
    labels = np.zeros((input_row, input_col))
    pixl_distance_t = []

    for i in range(input_row):
        for j in range(input_col):
            for k in range(len(center)):
                distance_t = np.sum(abs((input_signal[i, j]).astype(int) - center[k].astype(int))**2)
                pixl_distance_t.append(distance_t)
            labels[i, j] = int(pixl_distance_t.index(min(pixl_distance_t)))

            pixl_distance_t = []
    return labels


def pam(input_signal, center_num, threshold):
    '''
    基于pam算法的图像分割（适用于灰度图）
    :param input_signal:　输入图像
    :param center_num:　聚类中心数目
    :param threshold:　迭代阈值
    :return:
    '''
    input_signal_cp = np.copy(input_signal)
    input_row, input_col = input_signal_cp.shape
    labels = np.zeros((input_row, input_col))

    initial_center_row_num = [i for i in range(input_row)]
    random.shuffle(initial_center_row_num)
    initial_center_row_num = initial_center_row_num[:center_num]

    initial_center_col_num = [i for i in range(input_col)]
    random.shuffle(initial_center_col_num)

    present_center = []
    for i in range(center_num):
        present_center.append(input_signal_cp[initial_center_row_num[i], initial_center_row_num[i]])
    labels = classifier(input_signal_cp, present_center)

    num = 0
    while True:
        pre_centet = present_center.copy()  # 储存前一次的聚类中心
        # 计算当前聚类中心
        for n in range(center_num):
            temp = np.where(labels == n)
            present_center[n] = sum(input_signal_cp[temp].astype(int)) / len(input_signal_cp[temp])
        # 根据当前聚类中心分类
        labels = classifier(input_signal_cp, present_center)
        # 计算上一次聚类中心与当前聚类中心的差异
        loss = func_loss(present_center, pre_centet)
        num = num + 1
        print("Step:" + str(num) + "   Loss:" + str(loss))
        # 当损失小于迭代阈值时，结束迭代
        if loss <= threshold:
            break
    return labels


def main():
    '''
    需要提前运行salt.py，对img.jpg图像生成椒盐噪声
    '''
    path = 'img.jpg'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    i = 5  # 聚类中心数, 可设3,4,5,6
    new_img = pam(img, i, 1)
    new_img = cv2.normalize(new_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite('new_img{}.jpg'.format(i), new_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
