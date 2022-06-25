# by : Yangt
# -*- coding: utf-8 -*-
# 使用sklearn生成数据集进行聚类
import copy

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def distance(vectorA, vectorB):
    """
    计算两点之间的欧氏距离
    :param vectorA:
    :param vectorB:
    :return: 距离
    """
    temp = vectorA - vectorB
    return np.sqrt(np.dot(temp, temp.T))


def random_cents(dataSet, k):
    """
    随机抽取k行数据作为初始中心向量
    :param dataSet: 数据集
    :param k: 聚类数目
    :return: 初始中心向量
    """
    n = len(dataSet)
    cents_index = np.random.choice(n, k)
    # return [dataSet[index] for index in cents_index]
    return dataSet[cents_index]


def load_data():
    """
    加载数据
    :return:
    """
    data, label = make_blobs(n_samples=1000, n_features=2, centers=5)
    return data, label


class Kmeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cents = None
        self.C = None

    def fit(self, dataSet):
        """
        训练模型
        :param dataSet:训练数据集
        :return:
        """
        cents = random_cents(dataSet, self.n_clusters)
        cents = np.array(cents)
        C = []
        for i in range(self.max_iter):
            C = [[] for i in range(self.n_clusters)]
            for j in range(len(dataSet)):
                min_dis = float('inf')
                min_index = -1
                for k in range(self.n_clusters):
                    dis = distance(dataSet[j], cents[k])
                    if dis < min_dis:
                        min_dis = dis
                        min_index = k
                C[min_index].append(dataSet[j])

            C = [np.array(C[i]) for i in range(self.n_clusters)]

            temp = copy.deepcopy(cents)
            # print(id(cents), id(temp))
            # print(temp)
            for j in range(self.n_clusters):
                # print(C[j])
                cents[j] = np.sum(C[j], axis=0) / len(C[j])
                # print(np.sum(C[j], axis=0))
            # print(temp)
            # print(cents)
            # print(temp == cents)
            if (temp == cents).all():
                print(i)
                self.cents = cents[:]
                self.C = C[:]
                break
        self.cents = cents[:]
        self.C = C[:]

    def predict(self, dataSet):
        """判断未知数据属于哪一簇
        :param dataSet: 数据集
        :return: 每一条数据对应的簇索引
        """
        res = []
        for j in range(len(dataSet)):
            min_dis = float('inf')
            min_index = -1
            for k in range(self.n_clusters):
                dis = distance(dataSet[j], self.cents[k])
                if dis < min_dis:
                    min_dis = dis
                    min_index = k
            res.append(min_index)
        return res


if __name__ == '__main__':
    X, Y = load_data()
    km = Kmeans(5, 100)
    km.fit(X)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    f, axarr = plt.subplots(2, 1, figsize=(10, 8))
    Z = np.array(km.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    axarr[0].contourf(xx, yy, Z, alpha=0.3)  # contourf():画等高线，xx和yy为网格矩阵；Z为高度，及Z=f(x,y)；alpha参数表示颜色的深浅

    for k in range(km.n_clusters):
        axarr[0].scatter(km.C[k][:, 0], km.C[k][:, 1], s=20, edgecolor='k')

    # 画出中心向量
    axarr[0].scatter(km.cents[:, 0], km.cents[:, 1], c='r', marker='^', s=80)

    axarr[1].scatter(X[:, 0], X[:, 1], c=Y, s=20)
    plt.show()


