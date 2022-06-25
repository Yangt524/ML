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
        self.labelProperty = []

    def getDataSet(self, dataset, labels, labelProperty):
        self.dataSet = dataset
        self.labels = labels
        self.labelProperty = labelProperty

    def train(self):
        labels = self.labels[:]
        self.tree = self.buildTree(self.dataSet, labels, self.labelProperty)

    # 获取划分点属性对应的属性值集合
    def getUniqueFeatValues(self, bestFeatLabel):

        featValues = [ds[self.labels.index(bestFeatLabel)] for ds in self.dataSet]
        return set(featValues)

    # 构造决策树
    def buildTree(self, dataSet, labels, labelProperty):
        classList = [ds[-1] for ds in dataSet]
        if classList.count(classList[0]) == len(classList):
            return classList[0]
        if len(dataSet[0]) == 1:
            return self.classify(classList)

        bestFeat, bestPartValue = self.findBestSplit(dataSet, labelProperty)
        if labelProperty[bestFeat] == 0:
            bestFeatLabel = labels[bestFeat]
            tree = {bestFeatLabel: {}}
            del (labels[bestFeat])
            del (labelProperty[bestFeat])

            # featValues = [ds[bestFeat] for ds in dataSet]

            uniqueFeatValues = self.getUniqueFeatValues(bestFeatLabel)

            for value in uniqueFeatValues:
                subLabels = labels[:]
                subLabelProperty = labelProperty[:]
                subDataSet = self.splitDataSet(dataSet, bestFeat, value)
                if not subDataSet:
                    tree[bestFeatLabel][value] = self.classify(classList)
                else:
                    subTree = self.buildTree(subDataSet, subLabels, subLabelProperty)
                    tree[bestFeatLabel][value] = subTree
        else:
            bestFeatLabel = labels[bestFeat] + '<' + str(round(bestPartValue, 2))
            tree = {bestFeatLabel: {}}
            subLabels = labels[:]
            subLabelProperty = labelProperty[:]

            valueL = '是'
            subDataSetL = self.splitDataSet_c(dataSet, bestFeat, bestPartValue, 'L')
            subTree = self.buildTree(subDataSetL, subLabels, subLabelProperty)
            tree[bestFeatLabel][valueL] = subTree

            valueR = '否'
            subDataSetR = self.splitDataSet_c(dataSet, bestFeat, bestPartValue, 'R')
            subTree = self.buildTree(subDataSetR, subLabels, subLabelProperty)
            tree[bestFeatLabel][valueR] = subTree
        return tree

    # 分类，返回类别列表中占比更高的类别
    def classify(self, classList):
        items = dict([(classList.count(i), i) for i in classList])
        return items[max(items.keys())]

    # 通过计算信息增益，返回信息增益最大的属性，最优划分属性
    def findBestSplit(self, dataset, labelProperty):
        numFeatures = len(dataset[0])-1
        baseEntropy = self.calcShannonEnt(dataset)
        num = len(dataset)
        bestInfoGain = 0.0
        bestFeat = -1
        bestPartValue = None

        for i in range(numFeatures):
            featValues = [ds[i] for ds in dataset]
            # bestFeatLabel = labels[i]
            # uniqueFeatValues = self.getUniqueFeatValues(bestFeatLabel)
            uniqueFeatValues = set(featValues)
            newEntropy = 0.0
            bestPV = None

            if labelProperty[i] == 0:
                for val in uniqueFeatValues:
                    subDataSet = self.splitDataSet(dataset, i, val)
                    prob = len(subDataSet)/float(num)
                    newEntropy += prob*self.calcShannonEnt(subDataSet)
            else:
                sortedUniqueFeatV = list(uniqueFeatValues)
                sortedUniqueFeatV.sort()
                listPartition = []
                minEntropy = float('inf')

                for j in range(len(sortedUniqueFeatV)-1):
                    partValue = (float(sortedUniqueFeatV[j]) + float(sortedUniqueFeatV[j + 1])) / 2

                    dataSetLeft = self.splitDataSet_c(dataset, i, partValue, 'L')
                    dataSetRight = self.splitDataSet_c(dataset, i, partValue, 'R')
                    probLeft = len(dataSetLeft)/float(num)
                    probRight = len(dataset)/float(num)
                    Entropy = probLeft * self.calcShannonEnt(dataSetLeft) + probRight * self.calcShannonEnt(dataSetRight)
                    if Entropy < minEntropy:
                        minEntropy = Entropy
                        bestPV = partValue

                newEntropy = minEntropy

            infoGain = baseEntropy - newEntropy
            if infoGain > bestInfoGain:
                bestInfoGain = infoGain
                bestFeat = i
                bestPartValue = bestPV
        return bestFeat, bestPartValue

    # 划分数据集, axis:按第几个特征划分, value:划分特征的值, LorR: value值左侧（小于）或右侧（大于）的数据集
    def splitDataSet_c(self, dataSet, axis, value, LorR):
        retDataSet = []
        featVec = []
        if LorR == 'L':
            for featVec in dataSet:
                if float(featVec[axis]) < value:
                    retDataSet.append(featVec)
        else:
            for featVec in dataSet:
                if float(featVec[axis]) > value:
                    retDataSet.append(featVec)
        return retDataSet

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
        pass


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
        # features = ['Outlook', 'Temp', 'Humidity', 'Windy']

        # 西瓜数据集
        # dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.697, 0.460, '好瓜'],
        #            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.774, 0.376, '好瓜'],
        #            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.634, 0.264, '好瓜'],
        #            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 0.608, 0.318, '好瓜'],
        #            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 0.556, 0.215, '好瓜'],
        #            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.403, 0.237, '好瓜'],
        #            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', 0.481, 0.149, '好瓜'],
        #            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', 0.437, 0.211, '好瓜'],
        #            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', 0.666, 0.091, '坏瓜'],
        #            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', 0.243, 0.267, '坏瓜'],
        #            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', 0.245, 0.057, '坏瓜'],
        #            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', 0.343, 0.099, '坏瓜'],
        #            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', 0.639, 0.161, '坏瓜'],
        #            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', 0.657, 0.198, '坏瓜'],
        #            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 0.360, 0.370, '坏瓜'],
        #            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', 0.593, 0.042, '坏瓜'],
        #            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', 0.719, 0.103, '坏瓜']]
        # features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
        # labelProperty = [0, 0, 0, 0, 0, 0, 1, 1]

        # 鸢尾花数据集
        iris = datasets.load_iris()
        dataSet = iris.data.tolist()
        for i in range(len(dataSet)):
            dataSet[i].append(iris.target[i])
            # dataSet[i].append(iris.target_names[iris.target[i]])
        features = ['sl', 'sw', 'pl', 'pw']
        # features = ['sepal length', 'sepal width', 'petal length', 'petal width']
        labelProperty = [1, 1, 1, 1]
        return dataSet, features, labelProperty


    id3 = ID3Tree()
    ds, labels, labelProperty = createDataSet()
    id3.getDataSet(ds, labels, labelProperty)
    id3.train()
    print(id3.tree)
    # print(id3.predict(id3.tree, {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '沉闷', '纹理': '清晰', '脐部': '稍凹', '触感': '硬滑'}))
    treePlotter.createPlot(id3.tree)
