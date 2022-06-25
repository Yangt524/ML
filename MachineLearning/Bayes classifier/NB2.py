# by : Yangt
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from math import *

from sklearn import datasets
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


class NBayes:
    def __init__(self):
        self.priori_p = {}
        self.condition_p = {}

    def train(self, dataSet, targets, labelProperty):
        self.getPrioriP(targets)
        self.getConditionP(dataSet, targets, labelProperty)

    def getPrioriP(self, classList):
        num = len(classList)
        for cl in set(classList):
            self.priori_p[cl] = float(classList.count(cl)+1)/(num+len(set(classList)))      # 使用拉普拉斯修正
            # self.priori_p[cl] = float(classList.count(cl)) / num
        # print(self.priori_p)

    def splitDataSet(self, dataSet, targets, value):
        retDataSet = []
        for i in range(len(dataSet)):
            if targets[i] == value:
                retDataSet.append(dataSet[i])
        return retDataSet

    @staticmethod
    def getUniqueFeatValue(dataSet, i):
        featValue = [ds[i] for ds in dataSet]
        return set(featValue)

    def getConditionP(self, dataSet, targets, labelProperty):
        uniqueClass = set(targets)
        for value in uniqueClass:
            subDataSet = self.splitDataSet(dataSet, targets, value)
            for i in range(len(dataSet[-1])):
                subFeatValue = [ds[i] for ds in subDataSet]
                if labelProperty[i] == 0:
                    uniqueFeatValue = self.getUniqueFeatValue(dataSet, i)
                    N = len(uniqueFeatValue)

                    num = len(subDataSet)
                    for featV in uniqueFeatValue:
                        num_c = subFeatValue.count(featV)
                        self.condition_p[(featV, value)] = float((num_c+1))/(num+N)
                        # self.condition_p[(featV, value)] = float(num_c)/num
                else:
                    # mean = np.mean(subFeatValue)
                    # var = np.var(subFeatValue)
                    pass

    def predict(self, dataSet, targets, testData, labelProperty):
        uniqueClass = set(targets)
        p_c = {}
        for value in uniqueClass:
            subDataSet = self.splitDataSet(dataSet, targets, value)
            # p_l = log(self.priori_p[value])
            p = self.priori_p[value]
            for i in range(len(testData)):
                if labelProperty[i] == 0:
                    p *= self.condition_p[(testData[i], value)]
                else:
                    data_c = [ds[i] for ds in subDataSet]
                    mean = np.mean(data_c)
                    std = np.std(data_c)
                    p *= exp(-(testData[i] - mean) ** 2 / ((std ** 2) * 2)) / ((sqrt(2 * pi)) * std)
                    # p_test = log(exp(-pow(testData[i]-mean, 2)/(2*pow(std, 2)))/(sqrt(2*pi)*std))
            p_c[value] = p

        # print(p_c)
        return max(p_c, key=p_c.get)

    def display(self):
        print('先验概率:\n', self.priori_p)
        print('离散属性条件概率:\n', self.condition_p)


if __name__ == '__main__':
    def createDataSet():

        # 鸢尾花数据集
        # iris = datasets.load_iris()
        # dataSet = iris.data.tolist()
        # targets = iris.target.tolist()
        # features = iris.feature_names
        # labelProperty = [1, 1, 1, 1]

        # 乳腺癌数据集
        breast_cancer = datasets.load_breast_cancer()
        dataSet = breast_cancer.data.tolist()
        targets = breast_cancer.target.tolist()
        features = breast_cancer.feature_names
        labelProperty = [1] * 30
        return dataSet, targets, features, labelProperty


    nb = NBayes()
    dataSet, targets, features, labelProperty = createDataSet()
    datas_train, datas_test, target_train, target_test = train_test_split(dataSet, targets, test_size=0.3, random_state=123)
    nb.train(datas_train, target_train, labelProperty)
    nb.display()
    # res = nb.predict(dataSet, ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460], labelProperty)
    target_pred = []
    for dst in datas_test:
        res = nb.predict(datas_train, target_train, dst, labelProperty)
        target_pred.append(res)
    report = classification_report(target_test, target_pred, target_names=['malignant', 'benign'])

    # print(len(target_test))
    print('测试集结果及预测结果：')
    print(target_test)
    print(target_pred)
    l0_1 = []
    l1_0 = []
    for i in range(len(target_pred)):
        if target_test[i] == 0 and target_pred[i] == 1:
            l0_1.append(i)
        elif target_test[i] == 1 and target_pred[i] == 0:
            l1_0.append(i)

    print('将类别0错误预测为类别1的数据下标：')
    print(l0_1)
    print('将类别1错误预测为类别0的数据下标：')
    print(l1_0)
    print(report)

    # print(res)
