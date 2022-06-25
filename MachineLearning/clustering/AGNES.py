"""
@author:Yangt
@file:AGNES.py
@time:2021/11/23
@version:1.0
"""
import numpy as np
from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


np.random.seed(0)


def distance(vectorA, vectorB):
    """
    计算两点之间的欧氏距离
    :param vectorA:
    :param vectorB:
    :return: 距离
    """
    vectorA, vectorB = np.array(vectorA), np.array(vectorB)
    temp = vectorA - vectorB
    return np.sqrt((temp**2).sum())


def load_data():
    """
    加载数据
    Returns:

    """
    # X, Y = make_circles(n_samples=100, shuffle=True, noise=0.01)
    X, Y = make_blobs(n_samples=500, n_features=2, centers=5)

    # X = [list(x) for x in X]
    # Y = list(Y)
    return X, Y


def get_set_dist(A, B, dist_kind='max'):
    """
    获取集合A和B之间的距离
    Args:
        A:
        B:
        dist_kind: 集合距离的种类，三种选择：mean、min、max

    Returns:A，B之间的距离

    """
    dist = np.array([])
    for a in A:
        for b in B:
            d = distance(a, b)
            dist = np.append(dist, d)
    if dist_kind == 'min':
        return np.amin(dist, axis=0)
    elif dist_kind == 'max':
        return np.amax(dist, axis=0)
    elif dist_kind == 'mean':
        return np.mean(dist, axis=0)
    else:
        print('距离类型错误！\n')


def get_dist_matrix(C):
    """
    获取簇之间的距离矩阵
    Args:
        C: 当前簇划分

    Returns:距离矩阵

    """
    n = len(C)
    M = np.ones((n, n))
    for i in range(n):
        M[i][i] = float('inf')
        for j in range(i+1, n):
            M[i][j] = get_set_dist(C[i], C[j])
            M[j][i] = M[i][j]
    return M


def get_minIndex_matrix(Matrix):
    """
    获取矩阵中最小值的下标
    Args:
        Matrix: 矩阵

    Returns:
        r:最小值行坐标
        c:最小值列坐标

    """
    min_index = np.argmin(Matrix)
    r = int(min_index / len(Matrix))
    c = min_index % len(Matrix)
    return r, c


class AGNES:
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster
        self.C = None

    def fit(self, X):
        n = len(X)
        C = [[] for i in range(n)]
        # print(C)
        for i in range(n):
            # print(C[i], type(C[i]))
            C[i].append(X[i])

        M = get_dist_matrix(C)
        q = n
        while q > self.n_cluster:
            r, c = get_minIndex_matrix(M)
            C[r] = np.r_[C[r], C[c]]
            del C[c]
            # print(C)

            # 删除c行c列
            M = np.delete(M, c, axis=0)
            M = np.delete(M, c, axis=1)

            for j in range(q-1):
                if j != r:
                    M[r][j] = get_set_dist(C[r], C[j])
                    M[j][r] = M[r][j]
            q = q - 1
        self.C = C[:]

    def predict(self):
        pass


if __name__ == '__main__':
    X, Y = load_data()
    ag = AGNES(5)
    ag.fit(X)

    M = get_dist_matrix(ag.C)
    print(M)

    f, axarr = plt.subplots(2, 1)
    # color = ['r', 'g', 'b', 'y', 'm']
    # i = 0
    for c in ag.C:
        c = np.array(c)
        axarr[0].scatter(c[:, 0], c[:, 1], edgecolor='k', s=20)
        # i = i + 1
    axarr[1].scatter(X[:, 0], X[:, 1], c=Y, s=20, edgecolor='k')
    plt.show()
