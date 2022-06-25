# by : Yangt
# -*- coding: utf-8 -*-

import sympy
import numpy as np
import matplotlib.pyplot as plt


def get_a(_x_, _w_, p):
    var1 = var2 = 0
    for xi, wi in zip(_x_, _w_):
        px = sympy.simplify(p[-1]).subs(x, xi)
        var1 += wi * xi * px * px
        var2 += wi * px * px
    return float(var1)/var2


def get_b(_x_, _w_, p):
    var1 = var2 = 0
    for xi, wi in zip(_x_, _w_):
        px = sympy.simplify(p[-1]).subs(x, xi)
        px_1 = sympy.simplify(p[-2]).subs({x: xi})
        var1 += wi * px * px
        var2 += wi * px_1 * px_1
    return float(var1)/var2


def get_ak(_x_, _y_, _w_, p):
    ak = []
    for i in range(3):
        var1 = var2 = 0
        for xi, yi, wi in zip(_x_, _y_, _w_):
            px = sympy.simplify(p[i]).subs(x, xi)
            var1 += wi * yi * px
            var2 += wi * px * px
        ak.append(float(var1) / var2)
    return ak


def getPx(_x_, _w_, k, p, a, b):
    x = sympy.symbols('x')
    if k == 1:
        pi = x - a[-1]
    else:
        getPx(_x_, _w_, k-1, p, a, b)
        pi = p[-1] * (x - a[-1]) - b[-1] * p[-2]
    p.append(pi)
    a.append(get_a(_x_, _w_, p))
    b.append(get_b(_x_, _w_, p))
    return


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['SimHei']    # 显示中文

    _x_ = np.array([0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    _y_ = np.array([1.20, 1.50, 1.70, 2.00, 2.24, 2.40, 2.75, 3.00])
    _w_ = np.array([1, 1, 50, 1, 1, 1, 1, 1])
    plt.scatter(_x_, _y_, label=u'已知散点')       # 画出已知点的散点图

    '''
    # 提前计算出p[0]和a[0]，为后续计算做准备
    '''
    x = sympy.symbols('x')
    p0 = 0*x+1
    p = [p0]
    a = []
    b = []
    a.append(get_a(_x_, _w_, p))

    getPx(_x_, _w_, 2, p, a, b)     # 计算p_k(x)
    # print(p)

    ak = get_ak(_x_, _y_, _w_, p)   # 计算a_k
    # print(ak)

    sx = ak[0] * p[0] + ak[1] * p[1] + ak[2] * p[2]     # 求S(x)表达式
    sx_s = sympy.simplify(sx)   # 化简S(x)
    print('化简后的所求多项式为：', sx_s)

    # 画图
    px = np.linspace(0, 1, num=100)
    py = sympy.lambdify(x, sx_s, 'numpy')   # 将 SymPy 表达式转换为 NumPy 可使用的函数，从而计算出画图所需的纵坐标
    # print(py(px))
    plt.plot(px, py(px), 'r', label=u'拟合曲线')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='best')

    plt.show()
