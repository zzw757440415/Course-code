from math import *
import matplotlib.pyplot as plt
from pylab import *


# 绘制曲线，间隔为0.05
def draw(a, b , interp = 0.05):
    x = [a+ele*interp for ele in range(0, int((b-a)/interp))]
    y = [function(ele) for ele in x]
    plt.figure(1)
    plt.plot(x, y)
    xlim(a, b)
    title('3x^4-4x^3-12x^2', color="b")
    plt.show()


def function(x):
    fx = str_fx.replace("x", "%(x)f")
    #  带入x计算函数值
    return eval(fx % {"x": x})


# 黄金分割法进行一维搜索的函数
def gold_div_search(a , b, esp, label):
    data = list()
    x1 = a+rou*(b-a)
    x2 = b-rou*(b-a)
    data.append([a, x1, x2, b])
    while(b-a>esp):
        # 若f(x1)>f(x2)，则在区间(x1,b)内搜索
        if function(x1) > function(x2):
            a = x1
            x1 = x2
            x2 = b-rou*(b-a)
            plt.plot(x2, function(x2),'r*')
        # 如果f(x1)<f(x2),则在区间(a,x2)内搜索
        elif function(x1) < function(x2):
            b = x2
            x2 = x1
            x1 = rou*(b-a)
            plt.plot(x1,function(x1),'r*')
        # 如果f(x1)=f(x2)，则在区间(x1,x2)内搜索
        else:
            a = x1
            b = x2
            x1 = a+rou*(b-a)
            x2 = b-rou*(b-a)
            plt.plot(x1, function(x1),'r*',x2,function(x2),'r*')
        data.append([a,x1,x2,b])
    with open("黄金分割%d.txt" % label, mode="w", encoding="utf-8")as a_file:
        # 当前目录下保存结果
        for i in range(0, len(data)):
            a_file.write("%d：\t" % (i+1))
            for j in range(0, 4):
                a_file.write("function(%.3f)=%.3f\t"%(data[i][j],function(data[i][j])))
            a_file.write("\n")
    return [a,b]


rou = 1-(sqrt(5)-1)/2  # 1-rou为黄金分割比
str_fx = '3*x**4-4*x**3-12*x**2'
para = [-2, 0, 0.001]  # 导入区间
para = [float(ele) for ele in para]
a, b, esp = para
gold_div_search(a, b, esp, 1)  # 调用黄金分割法并保存文件
draw(a, b, (b-a)/2000)  # 绘制函数图形

para = [0, 3, 0.001]
para = [float(ele) for ele in para]
a, b, esp = para
gold_div_search(a, b, esp, 2)
draw(a, b, (b-a)/2000)





