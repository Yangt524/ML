# by : Yangt
# -*- coding: utf-8 -*-

import copy

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


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
    :return: 初始中心向量在数据集中的索引
    """
    n = len(dataSet)
    cents_index = np.random.choice(n, k)
    return cents_index


def load_data():
    """
    加载数据
    :return:
    """
    iris = datasets.load_iris()
    X = iris.data[:, 0:2]
    return X


class Kmeans:
    def __init__(self, n_clusters, max_iter):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cents = None
        self.C = None

    def fit(self, dataSet):
        """
        训练模型
        :param dataSet:
        :return:
        """
        cents_index = random_cents(dataSet, self.n_clusters)
        cents = [dataSet[index] for index in cents_index]
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
            cents = np.array(cents)
            temp = copy.deepcopy(cents)

            for j in range(self.n_clusters):
                cents[j] = np.sum(C[j], axis=0) / len(C[j])

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


def main():
    X = load_data()
    km = Kmeans(3, 100)
    km.fit(X)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # 生成坐标矩阵
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    f, axarr = plt.subplots(1, figsize=(10, 8))
    Z = np.array(km.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    axarr.contourf(xx, yy, Z, alpha=0.3)  # contourf():画等高线，xx和yy为网格矩阵；Z为高度，及Z=f(x,y)；alpha参数表示颜色的深浅

    for k in range(km.n_clusters):
        axarr.scatter(km.C[k][:, 0], km.C[k][:, 1], s=20, edgecolor='k')

    # 画出中心向量
    axarr.scatter(km.cents[:, 0], km.cents[:, 1], c='r', marker='^', s=80)
    plt.show()


if __name__ == '__main__':
    main()


