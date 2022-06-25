"""
@author:Yangt
@file:Mixture_of_Gaussian.py
@time:2021/11/21
@version:1.0
"""
import numpy as np
from sklearn.datasets import make_blobs


def distance(vectorA, vectorB):
    """
    计算两点之间的欧氏距离
    :param vectorA:
    :param vectorB:
    :return: 距离
    """
    # vectorA, vectorB = np.array(vectorA), np.array(vectorB)
    temp = vectorA - vectorB
    return np.sqrt((temp**2).sum())


def load_data():
    """
    加载数据
    Returns:

    """
    X, Y = make_blobs(n_samples=1000, n_features=2, centers=5)
    return X, Y


class MixtureOfGaussian:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


if __name__ == '__main__':
    pass
