# by : Yangt
# -*- coding: utf-8 -*-
from math import log2

from sklearn import datasets

import treePlotter


class ID3Tree(object):
    def __init__(self):
        self.tree = {}
        self.dataSet = []
        self.labels = []

    def getDataSet(self, dataset, labels):
        self.dataSet = dataset
        self.labels = labels

    def train(self):
        labels = self.labels[:]
        self.tree = self.buildTree(self.dataSet, labels)

    # 获取划分点属性对应的属性值集合
    def getUniqueFeatValues(self, bestFeatLabel):

        featValues = [ds[self.labels.index(bestFeatLabel)] for ds in self.dataSet]
        return set(featValues)

    # 构造决策树
    def buildTree(self, dataSet, labels):
        classList = [ds[-1] for ds in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.classify(classList)

        bestFeat = self.findBestSplit(dataSet)
        bestFeatLabel = labels[bestFeat]
        tree = {bestFeatLabel: {}}
        del (labels[bestFeat])

        # featValues = [ds[bestFeat] for ds in dataSet]

        uniqueFeatValues = self.getUniqueFeatValues(bestFeatLabel)

        for value in uniqueFeatValues:
            subLabels = labels[:]
            subDataSet = self.splitDataSet(dataSet, bestFeat, value)
            if subDataSet == []:
                tree[bestFeatLabel][value] = self.classify(classList)
            else:
                subTree = self.buildTree(subDataSet, subLabels)
                tree[bestFeatLabel][value] = subTree
        return tree

    # 分类，返回类别列表中占比更高的类别
    def classify(self, classList):
        items = dict([(classList.count(i), i) for i in classList])
        return items[max(items.keys())]

    # 通过计算信息增益，返回信息增益最大的属性，最优划分属性
    def findBestSplit(self, dataset):
        numFeatures = len(dataset[0])-1
        baseEntropy = self.calcShannonEnt(dataset)
        num = len(dataset)
        bestInfoGain = 0.0
        bestFeat = -1

        for i in range(numFeatures):
            featValues = [ds[i] for ds in dataset]
            # bestFeatLabel = labels[i]
            # uniqueFeatValues = self.getUniqueFeatValues(bestFeatLabel)
            uniqueFeatValues = set(featValues)
            newEntropy = 0.0

            for val in uniqueFeatValues:
                subDataSet = self.splitDataSet(dataset, i, val)
                prob = len(subDataSet)/float(num)
                newEntropy += prob*self.calcShannonEnt(subDataSet)
            infoGain = baseEntropy-newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeat = i
        return bestFeat

    # 划分子集
    def splitDataSet(self, dataset, feat, values):
        retDataSet = []
        for featVec in dataset:
            if featVec[feat] == values:
                reducedFeatVec = featVec[:feat]
                reducedFeatVec.extend(featVec[feat+1:])
                retDataSet.append(reducedFeatVec)
        return retDataSet

    # 计算香农熵
    def calcShannonEnt(self, dataSet):
        num = len(dataSet)
        classList = [c[-1] for c in dataSet]
        labelCounts = {}
        for cs in set(classList):
            labelCounts[cs] = classList.count(cs)

        shannonEnt = 0.0
        for key in labelCounts:
            prob = labelCounts[key]/float(num)
            shannonEnt -= prob*log2(prob)
        return shannonEnt

    # 利用生成的决策树预测新数据所属类别
    def predict(self, tree, newObject):
        while type(tree).__name__ == 'dict':
            key = list(tree.keys())[0]
            tree = tree[key][newObject[key]]
        return tree


if __name__ == '__main__':
    def createDataSet():
        # dataSet = [[2, 1, 0, 1, 'No'],
        #            [2, 1, 0, 0, 'No'],
        #            [0, 1, 0, 1, 'Yes'],
        #            [1, 0, 1, 1, 'Yes'],
        #            [1, 2, 0, 1, 'Yes'],
        #            [1, 0, 1, 0, 'No'],
        #            [0, 0, 1, 0, 'Yes'],
        #            [2, 2, 0, 1, 'No'],
        #            [2, 0, 1, 1, 'Yes'],
        #            [1, 2, 1, 1, 'Yes'],
        #            [2, 2, 1, 0, 'Yes'],
        #            [0, 2, 0, 0, 'Yes'],
        #            [0, 1, 1, 1, 'Yes'],
        #            [1, 2, 0, 0, 'No']]
        # feateres = ['Outlook', 'Temp', 'Humidity', 'Windy']

        dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                   ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
                   ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                   ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
                   ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
                   ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
                   ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
                   ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
                   ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
                   ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
                   ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
                   ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
                   ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
                   ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
                   ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
                   ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
                   ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']]
        features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        # iris = datasets.load_iris()
        # dataSet = iris.data.tolist()
        # for i in range(len(dataSet)):
        #     dataSet[i].append(iris.target[i])
        #
        # features = ['a', 'b', 'c', 'd']
        return dataSet, features


    id3 = ID3Tree()
    ds, labels = createDataSet()
    id3.getDataSet(ds, labels)
    id3.train()
    print(id3.tree)
    # print(id3.predict(id3.tree, {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '沉闷', '纹理': '清晰', '脐部': '稍凹', '触感': '硬滑'}))
    treePlotter.createPlot(id3.tree)
