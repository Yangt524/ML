# by : Yangt
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


def getGD(x, y, w):

    """
    :param x: 已知点横坐标数组
    :param y: 已知点纵坐标数组
    :param w: 已知点权值数组
    :return: 法方程系数矩阵g及方程右侧值d
    """
    g = []
    d = []
    for i in range(3):
        g_c = []
        for j in range(3):
            var = 0
            for xi, wi in zip(x, w):
                var += wi * xi ** i * xi ** j
            g_c.append(var)

        g.append(g_c)

        var_d = 0
        for xi, yi, wi in zip(x, y, w):
            var_d += wi * yi * xi**i
        d.append(var_d)

    return g, d


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['SimHei']  # 显示中文

    x_i = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_i = np.array([1.20, 1.50, 1.70, 2.00, 2.24, 2.40, 2.75, 3.00])
    w_i = np.array([1, 1, 50, 1, 1, 1, 1, 1])
    plt.scatter(x_i, y_i, label=u'已知散点')

    g, d = getGD(x_i, y_i, w_i)

    result = np.dot(np.linalg.inv(g), d)    # 使用法方程系数矩阵进行求解，即g与d做内积运算

    x = np.linspace(0, 1, num=100)
    y = result[0] + result[1] * x + result[2] * x ** 2
    # print(result)
    print('所求多项式为：%.2f + %.2f * x + %.2f * x**2' % (result[0], result[1], result[2]))

    plt.plot(x, y, 'r', label=u'拟合曲线')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')
    plt.show()
