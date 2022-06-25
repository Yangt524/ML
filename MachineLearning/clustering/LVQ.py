"""
@author:Yangt
@file:LVQ.py
@time:2021/11/21
@version:1.0
"""
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


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


def get_cents(dataSet, cents_label):
    """
    获取初始原型向量
    Args:
        dataSet: 训练集，每行数据包括特征和类别标记
        cents_label: 每一个簇所对应的类别标记

    Returns:

    """
    cents = []
    n = len(dataSet)
    for cl in cents_label:
        while 1:
            i = np.random.choice(n)
            if dataSet[i][-1] == cl:
                cents.append(dataSet[i])
                break
    return cents


class LVQ:
    def __init__(self, n_clusters, cents_label):
        self.n_clusters = n_clusters
        self.cents_label = cents_label
        self.cents = None

    def fit(self, dataSet, learning_rate=0.1):
        """
        训练数据，主要是训练出原型向量
        Args:
            dataSet: 数据集，每一行数据包括特征和类别标记
            learning_rate:学习率

        Returns:None

        """
        cents = get_cents(dataSet, self.cents_label)
        for i in range(100):
            x_index = np.random.choice(len(dataSet))
            x = dataSet[x_index]
            dist = []
            for cent in cents:
                dist.append(distance(x[0: -1], cent[0: -1]))
            index = np.argmin(dist)
            if x[-1] == cents_label[index]:
                cents[index][0: -1] = cents[index][0: -1] + learning_rate * (x[0: -1] - cents[index][0: -1])
            else:
                cents[index][0: -1] = cents[index][0: -1] - learning_rate * (x[0: -1] - cents[index][0: -1])

        self.cents = np.array(cents)

    def predict(self, X):
        """
        判断数据属于哪个簇
        Args:
            X:待分类的数据集，每一行数据仅包含特征

        Returns:簇标记列表

        """
        labels = []
        for x in X:
            dist = []
            for cent in self.cents:
                dist.append(distance(x, cent[0: -1]))
            index = np.argmin(dist)
            labels.append(index)
        return labels


if __name__ == '__main__':
    X, Y = load_data()
    dataSet = np.c_[X, Y]
    cents_label = [0, 1, 2, 3, 4, 3, 4]
    lvq = LVQ(7, cents_label)
    lvq.fit(dataSet)
    res = lvq.predict(X)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 1, figsize=(10, 8))
    Z = np.array(lvq.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    axarr[0].contourf(xx, yy, Z, alpha=0.3)  # contourf():画等高线，xx和yy为网格矩阵；Z为高度，及Z=f(x,y)；alpha参数表示颜色的深浅


    axarr[0].scatter(X[:, 0], X[:, 1], c=res, s=20, edgecolor='k')

    # 画出中心向量
    axarr[0].scatter(lvq.cents[:, 0], lvq.cents[:, 1], c='r', marker='^', s=80)

    axarr[1].scatter(X[:, 0], X[:, 1], c=Y, s=20)
    plt.show()
