# by : Yangt
# -*- coding: utf-8 -*-
import numpy as np
from sklearn import datasets
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd


np.random.seed(0)


def sign(x):
    for i in range(len(x)):
        if x[i] >= 0:
            x[i] = 1
        else:
            x[i] = -1
    return x


class AdaBoosting:
    def __init__(self, clf_num=50):
        self.clfs = []
        self.alphas = []
        self.clf_num = clf_num

    def fit(self, X, Y):
        w = np.ones(len(Y))/len(Y)
        for i in range(self.clf_num):
            clf = DecisionTreeClassifier(max_depth=2)
            clf.fit(X, Y, w)
            Y_pred = clf.predict(X)
            e = 0.0
            # print(Y)
            # print(Y_pred)
            for j in range(len(Y)):
                if Y[j] != Y_pred[j]:
                    e += w[j]
            if e > 0.5:
                break
            alpha = 0.5 * np.log((1-e)/e)
            Z = np.dot(w, np.exp(-alpha * Y * Y_pred).T)
            w = (w/Z) * np.exp(-alpha * Y * Y_pred)
            self.clfs.append(clf)
            self.alphas.append(alpha)

    def predict(self, X):
        # fx = 0.0
        Y_pred = [clf.predict(X) for clf in self.clfs]
        fx = np.dot(np.array(self.alphas), np.array(Y_pred))
        # for clf, alpha in zip(self.clfs, self.alphas):
        #     y_p = clf.predict(X)d
        #     fx += alpha * y_p

        return sign(fx[:])


def LoadData():
    # iris = datasets.load_breast_cancer()
    # X = iris.data
    # Y = iris.target
    # for i in range(len(Y)):
    #     if Y[i] == 0:
    #         Y[i] = -1
    #
    # temp = list(zip(X, Y))
    # np.random.shuffle(temp)
    # X, Y = zip(*temp)
    # X, Y = np.array(X), np.array(Y)

    # 生成数据集
    X, Y = make_hastie_10_2(n_samples=30000, random_state=1)
    # 保存数据集
    # df = np.c_[X, Y]
    # df = pd.DataFrame(df)
    # # print(df)
    # df.to_csv('dateset.csv')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=123)
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = LoadData()
    clf = DecisionTreeClassifier(max_depth=2)
    ada = AdaBoosting()

    clf.fit(X_train, Y_train)
    ada.fit(X_train, Y_train)

    Y_pred_tree = clf.predict(X_test)
    Y_pred = ada.predict(X_test)
    # print(Y_pred)
    # print(Y_test)

    report_tree = classification_report(Y_test, Y_pred_tree)
    report = classification_report(Y_test, Y_pred)

    print('DT：\n', report_tree)
    print('AdaBoosting：\n', report)
