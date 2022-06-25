"""
@author:Yangt
@file:MDS.py
@time:2021/11/27
@version:1.0
"""


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.datasets import make_blobs

datasets.load_iris()

def load_data():
    """
    加载数据
    Returns:

    """

    # X, Y = make_circles(n_samples=100, shuffle=True, noise=0.01)
    X, Y = make_blobs(n_samples=2000, n_features=3, centers=5)

    # X = [list(x) for x in X]
    # Y = list(Y)
    return X, Y


def distance(vectorA, vectorB):
    """
    计算两点之间的欧氏距离
    :param vectorA:
    :param vectorB:
    :return: 距离
    """
    vectorA, vectorB = np.array(vectorA), np.array(vectorB)
    temp = vectorA - vectorB
    return np.sqrt(np.sum(temp**2))


def get_distance_matrix(X):
    n = len(X)
    M = np.ones((n, n))
    for i in range(n):
        M[i][i] = 0
        for j in range(i+1, n):
            M[i][j] = distance(X[i], X[j])
            M[j][i] = M[i][j]

    return M


class MDS:
    @staticmethod
    def dimension_reduction(X, d_):

        n = len(X)
        M = get_distance_matrix(X)
        dist_i_square = np.sum(M, axis=0) / n
        dist_j_square = np.sum(M, axis=1) / n
        dist_square = np.sum(M) / (n**2)

        B = np.ones((n, n))
        for i in range(n):
            for j in range(i, n):
                B[i][j] = -((M[i][j] ** 2) - dist_i_square[i] - dist_j_square[j] + dist_square) / 2
                B[j][i] = B[i][j]
        # print(B)

        val, vec = np.linalg.eig(B)
        val, vec = np.real(val), np.real(vec)
        index = np.argsort(-val)[range(d_)]
        val_ = val[index]
        vec_ = vec[:, index]
        # print(np.sqrt(val_))
        # print(vec_)
        return np.dot(vec_, np.diag(np.sqrt(val_)))


if __name__ == '__main__':
    X, Y = load_data()
    X_ = MDS.dimension_reduction(X, 2)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, edgecolor='k')
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_[:, 0], X_[:, 1], c=Y, edgecolor='k')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()


