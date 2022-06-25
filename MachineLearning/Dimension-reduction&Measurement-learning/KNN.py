"""
@author:Yangt
@file:KNN.py
@time:2021/11/19
@version:1.0
"""
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from collections import Counter


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
    :return:
    """
    # iris = datasets.load_breast_cancer()
    # X = iris.data
    # Y = iris.target
    # temp = list(zip(X, Y))
    # np.random.shuffle(temp)
    # X, Y = zip(*temp)
    # X, Y = np.array(X), np.array(Y)
    X, Y = make_hastie_10_2(n_samples=3000, random_state=1)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    return X_train, X_test, Y_train, Y_test


class KNN:
    def __init__(self, k):
        self.dataSet = []
        self.target = []
        self.k = k

    def fit(self, X, Y):
        """
        训练模型
        :param X:
        :param Y:
        """
        self.dataSet = X
        self.target = Y

    def predict(self, X):
        """
        预测
        :param X: 测试集
        :return: 预测结果
        """
        target = []

        for x in X:
            dist = np.array([])
            for i in range(len(self.target)):
                d = distance(x, self.dataSet[i])
                dist = np.append(dist, d)
            index = np.argsort(dist)[0:self.k]
            res = self.target[index]
            items = Counter(res)
            ts = max(items, key=items.get)
            target.append(ts)
        return target


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = load_data()
    knn = KNN(11)
    knn.fit(X_train, Y_train)
    res = knn.predict(X_test)
    # print(res)
    report = classification_report(Y_test, res)
    print(report)
