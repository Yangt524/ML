# by : Yangt
# -*- coding: utf-8 -*-
import numpy as np
from math import *

from sklearn import datasets


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

    def getUniqueFeatValue(self, dataSet, i):
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
                    sum = 0
                    for dc in data_c:
                        sum += (dc-mean)**2
                    var = sum / (len(data_c)-1)
                    # std = np.std(data_c)
                    std = sqrt(var)
                    p_ttt = exp(-(testData[i] - mean) ** 2 / ((std ** 2) * 2)) / ((sqrt(2 * pi)) * std)
                    print(p_ttt)
                    p *= p_ttt
                    # p_test = log(exp(-pow(testData[i]-mean, 2)/(2*pow(std, 2)))/(sqrt(2*pi)*std))
            p_c[value] = p

        print(p_c)
        return max(p_c, key=p_c.get)

    def display(self):
        print(self.priori_p)
        print(self.condition_p)


if __name__ == '__main__':
    def createDataSet():
        dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460],
                   ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376],
                   ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264],
                   ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318],
                   ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215],
                   ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237],
                   ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149],
                   ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211],
                   ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091],
                   ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267],
                   ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057],
                   ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099],
                   ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161],
                   ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198],
                   ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370],
                   ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042],
                   ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103]]
        targets = ['好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '好瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜', '坏瓜']
        features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
        labelProperty = [0, 0, 0, 0, 0, 0, 1, 1]

        # # 鸢尾花数据集
        # iris = datasets.load_iris()
        # dataSet = iris.data.tolist()
        # targets = iris.target.tolist()
        # # for i in range(len(dataSet)):
        # #     targets.append(iris.target_names[iris.target[i]])
        # features = ['sl', 'sw', 'pl', 'pw']
        # # features = ['sepal length', 'sepal width', 'petal length', 'petal width']
        # labelProperty = [1, 1, 1, 1]
        return dataSet, targets, features, labelProperty


    nb = NBayes()
    dataSet, targets, features, labelProperty = createDataSet()
    nb.train(dataSet, targets, labelProperty)
    nb.display()
    res = nb.predict(dataSet, targets, ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460], labelProperty)
    # res = nb.predict(dataSet, targets, [5.1, 3.5, 1.4, 0.2], labelProperty)
    print(res)
