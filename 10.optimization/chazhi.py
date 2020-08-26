# import numpy as np
# import matplotlib.pyplot as plt
#
# h = 0.0001
# l = np.linspace(-3, 5, 100)
# func = lambda x: 3 * x ** 4 - 4 * x ** 3 - 12 * x ** 2
#
#
# def function(x,f=func):  # 求函数在x处的导数
#     return (f(x+h)-f(x))/h
#
#
# def Newton(x, count):
#     for i in range(count):
#         x = x - function(x)/function(x,function)  #
#         print(i+1, '\t', x)
#
#
# # 输出迭代次数和此时的x值
# Newton(2.5, 10)
#
# -*- coding: utf-8 -*-

# '''
# 二次插值法python实现
# f(x)=x^4 - 4x^3 - 6x^2 -16x +4极值
# 区间[-1,6] e=0.05
# https://wenku.baidu.com/view/971f4116cd7931b765ce0508763231126edb773c.html
# '''
# import numpy as np
# import matplotlib.pyplot as plt
# from time import sleep
# from threading import Thread
#
# '''
# 函数表达式
# '''
#
#
# def f(x):
#     # return 1.0 * (pow(x, 4) - 4 * pow(x, 3) - 6 * pow(x, 2) - 16 * x + 4)
#     return 3 * pow(x, 4) - 4 * pow(x, 3) - 12 * pow(x, 2)
#
#
# # 定义变量们
# X2, Y = list(), list()
# k = 0
# a = -2.5  # 左点
# b = -0.3  # 右点
# e = 0.001  # 精度
# '''
# 绘制函数图像
# '''
#
#
# def close(time=1):
#     sleep(time)
#     # plt.savefig('./img/'+name)
#     plt.close()
#     pass
#
#
# def printFunc():
#     t = np.arange(a, b, 0.01)
#     s = f(t)
#     plt.plot(t, s)
#
#
# def update_point(x, y):
#     global k
#     printFunc()
#
#     plt.plot(x, y, 'ro')
#     plt.text(x[-1], y[-1], k, color='red', fontsize=k + 10)
#     # else:
#     #     plt.plot([x], [y], 'ro')
#     #     plt.text(x, y, k, color='red', fontsize=k + 10)
#     thread1 = Thread(target=close, args=())
#     thread1.start()
#     # print('打开')
#     plt.show()
#     # print("close")
#
#
# def final_fun(x, y):
#     global k
#     printFunc()
#     plt.plot(x, y, 'ro')
#     for i in range(1, k + 1):
#         plt.text(x[i - 1], y[i - 1], i, color='red', fontsize=i + 10)
#     # thread1 = Thread(target=close, args=())
#     # thread1.start()
#     plt.show()
#
#
# '''
# e为精度
# '''
#
#
# def search(f, x1, x2, x3):
#     global k
#     k += 1
#     if f(x2) > f(x1) or f(x2) > f(x3):
#         print("不满足两头大中间小的性质")
#         return 0
#
#     # 系数矩阵
#     A = [[pow(x1, 2), x1, 1], [pow(x2, 2), x2, 1], [pow(x3, 2), x3, 1]]
#     b = [f(x1), f(x2), f(x3)]
#
#     X = np.linalg.solve(A, b)
#
#     a0, a1, _ = X
#
#     x = - a1 / (2 * a0)
#
#     # 达到精度退出
#     if abs(x - x2) < e:
#         if f(x) < f(x2):
#             y = f(x)
#             print('最后的x:', x)
#             X2.append(x)
#             Y.append(y)
#             final_fun(X2, Y)
#             return (X2, Y)
#         else:
#             y = f(x2)
#             print("最后的x2", x2)
#             X2.append(x2)
#             Y.append(y)
#             final_fun(X2, Y)
#             return (X2, Y)
#     arr = [x1, x2, x3, x]
#     arr.sort()
#     # 在x2和新算出的x中找最小值
#     if f(x2) > f(x):
#         index = arr.index(x)
#         x2 = x
#     else:
#         index = arr.index(x2)
#
#     x1 = arr[index - 1]
#     x3 = arr[index + 1]
#     X2.append(x2)
#     Y.append(f(x2))
#     print('运行中的第%d次：%f' % (k, x2))
#     update_point(X2, Y)
#
#     return search(f, x1, x2, x3)
#
#
# def regre(f, a, b):
#     x1 = a
#     x3 = b
#     x2 = (a + b) / 2.0
#     search(f, x1, x2, x3)
#
#
# regre(f, a, b)


import numpy as np
import matplotlib.pyplot as plt
import math


def phi(x):
    '''
        测试函数1
    :param x:
    :return:
    '''
    # return x * x - 2 * x + 1
    return 3*x**4-4*x**3-12*x**2


def complicated_func(x):
    '''
        测试函数2
    :param x:
    :return:
    '''
    # return x * x * x + 5 * math.sin(2 * x)
    return 3 * x ** 4 - 4 * x ** 3 - 12 * x ** 2


def parabolic_search(f, a, b, epsilon=1e-1):
    '''
        抛物线法，迭代函数
    :param f: 目标函数
    :param a:   起始点
    :param b:   终止点
    :param epsilon: 阈值
    :return:
    '''
    h = (b - a) / 2
    s0 = a
    s1 = a + h
    s2 = b
    f0 = f(s0)
    f1 = f(s1)
    f2 = f(s2)
    h_mean = (4 * f1 - 3 * f0 - f2) / (2 * (2 * f1 - f0 - f2)) * h
    s_mean = s0 + h_mean
    f_mean = f(s_mean)
    # 调试
    k = 0
    while s2 - s0 > epsilon:
        h = (s2 - s0) / 2
        h_mean = (4 * f1 - 3 * f0 - f2) / (2 * (2 * f1 - f0 - f2)) * h
        s_mean = s0 + h_mean
        f_mean = f(s_mean)
        if f1 <= f_mean:
            if s1 < s_mean:
                s2 = s_mean
                f2 = f_mean
                # 重新计算一次，书上并没有写，所以导致一直循环
                s1 = (s2 + s0)/2
                f1 = f(s1)
            else:
                s0 = s_mean
                f0 = f_mean
                s1 = (s2 + s0)/2
                f1 = f(s1)
        else:
            if s1 > s_mean:
                s2 = s1
                s1 = s_mean
                f2 = f1
                f1 = f_mean
            else:
                s0 = s1
                s1 = s_mean
                f0 = f1
                f1 = f_mean
        # print([k, (s2 - s0), f_mean, s_mean])
        print(k)
        k += 1
    return s_mean, f_mean


if __name__ == '__main__':
    x = np.linspace(1, 3, 200)
    y = []
    index = 0
    for i in x:
        y.append(complicated_func(x[index]))
        index += 1
    plt.plot(x, y)
    plt.show()

    result = parabolic_search(complicated_func, 1.0, 3.0)
    print(result)

    # x = np.linspace(0, 2, 200)
    # plt.plot(x, phi(x))
    # plt.show()
    # result = parabolic_search(phi, 0, 2.0)
    # print(result)

