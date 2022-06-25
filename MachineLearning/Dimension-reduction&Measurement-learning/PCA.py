"""
@author:Yangt
@file:PCA.py
@time:2021/11/29
@version:
"""
import copy

import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def load_data():
    """
    加载数据
    Returns:

    """
    # X, Y = make_circles(n_samples=100, shuffle=True, noise=0.01)
    X, Y = make_blobs(n_samples=500, n_features=3, centers=5)

    # X = [list(x) for x in X]
    # Y = list(Y)
    return X, Y


def Centralization(X):
    for i in range(len(X)):
        X[i] = X[i] - np.mean(X[i])
    return X


class PCA:
    def __init__(self):
        self.W = None

    def dimension_reduction(self, X, d_):
        X = X - np.mean(X, axis=0)
        # print(np.var(X, axis=0))
        # Centralization(X)
        Cov = np.cov(X.T)
        print(Cov)
        print(np.dot(X.T, X)/len(X))

        val, vec = np.linalg.eig(Cov)
        val, vec = np.real(val), np.real(vec)
        index = np.argsort(-val)[range(d_)]
        # val_ = val[index]
        vec_ = vec[:, index]
        self.W = vec_[:]
        print(self.W)
        return np.dot(vec_.T, X.T).T


if __name__ == '__main__':
    X, Y = load_data()
    _X_ = Centralization(copy.deepcopy(X))
    pca = PCA()
    X_ = pca.dimension_reduction(_X_, 2)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=Y, edgecolor='k')
    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.scatter3D(X_[:, 0], X_[:, 1], X_[:, 2], c=Y, edgecolor='k')
    ax2 = fig.add_subplot(122)
    ax2.scatter(X_[:, 0], X_[:, 1], c=Y, edgecolor='k')
    fig.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()
