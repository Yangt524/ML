# by : Yangt
# -*- coding: utf-8 -*-
from math import log2

from sklearn import datasets

import treePlotter
from numpy import *


# 获取划分点属性对应的属性值集合
def getUniqueFeatValues(bestFeatLabel):
    # from C45Tree import main
    global __dataSet__, __labels__
    featValues = [ds[__labels__.index(bestFeatLabel)] for ds in __dataSet__]
    return set(featValues)


# 计算香农熵
def calcShannonEnt(dataSet):
    num = len(dataSet)
    classList = [c[-1] for c in dataSet]
    labelCounts = {}
    for cs in set(classList):
        labelCounts[cs] = classList.count(cs)

    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / float(num)
        shannonEnt -= prob * log2(prob)
    return shannonEnt


# 划分子集
def splitDataSet(dataset, feat, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[feat] == value:
            reducedFeatVec = featVec[:feat]
            reducedFeatVec.extend(featVec[feat+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


def calcSplitInfo(featureVList):
    numEntries = len(featureVList)
    featureValueSetList = list(set(featureVList))
    valueCounts = [featureVList.count(featVect) for featVect in featureValueSetList]

    pList = [float(item) / numEntries for item in valueCounts]
    iList = [item * math.log2(item) for item in pList]
    splitInfo = -sum(iList)
    return splitInfo, featureValueSetList


def getBestFeat(dataSet):
    numFeats = len(dataSet[0])-1
    num = len(dataSet)

    newEntropy = calcShannonEnt(dataSet)
    ConditionEntropy = []
    splitInfo = []
    allFeatVList = []

    for fe in range(numFeats):
        featList = [data[fe] for data in dataSet]
        splitI, featureValueList = calcSplitInfo(featList)
        allFeatVList.append(featureValueList)
        splitInfo.append(splitI)
        resultGain = 0.0

        for value in featureValueList:
            subSet = splitDataSet(dataSet, fe, value)
            appearNum = float(len(subSet))
            subEntropy = calcShannonEnt(subSet)
            resultGain += (appearNum / num)*subEntropy
        ConditionEntropy.append(resultGain)

    infoGainArray = newEntropy * ones(numFeats) - array(ConditionEntropy)
    infoGainRatio = infoGainArray / array(splitInfo)
    bestFeatureIndex = argsort(-infoGainRatio)[0]
    return bestFeatureIndex, allFeatVList[bestFeatureIndex]


def majorityCnt(classList):
    items = dict([(classList.count(i), i) for i in classList])
    return items[max(items.keys())]


def C45Tree(dataSet, labels):
    classList = [ds[-1] for ds in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat, featValueList = getBestFeat(dataSet)

    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])

    uniqueFeatValues = getUniqueFeatValues(bestFeatLabel)

    for value in uniqueFeatValues:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value)
        if subDataSet == []:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
        else:
            myTree[bestFeatLabel][value] = C45Tree(subDataSet, subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    """
    :param inputTree:
    :param featLabels:
    :param testVec:
    :return: 决策结果
    """
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)

    classLabel = []
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def predict(inputTree, featLabels, testDataSet):
    """
    :param inputTree:
    :param featLabels:
    :param testDataSet:
    :return: 决策结果
    """
    classLabelAll = []
    for testVec in testDataSet:
        classLabelAll.append(classify(inputTree, featLabels, testVec))
    return classLabelAll


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
    # labels = ['Outlook', 'Temp', 'Humidity', 'Windy']

    # dataSet = [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    #            ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    #            ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    #            ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
    #            ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
    #            ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
    #            ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
    #            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
    #            ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
    #            ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
    #            ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
    #            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
    #            ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
    #            ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
    #            ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
    #            ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
    #            ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']]
    # labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    iris = datasets.load_iris()
    dataSet = iris.data.tolist()
    for i in range(len(dataSet)):
        dataSet[i].append(iris.target[i])

    labels = ['a', 'b', 'c', 'd']
    return dataSet, labels


def createTestSet():
    testSet = [[0, 1, 0, 0],
               [0, 2, 1, 0],
               [2, 1, 1, 0],
               [0, 1, 1, 1],
               [1, 1, 0, 1],
               [1, 0, 1, 0],
               [2, 1, 0, 1]]
    return testSet


__dataSet__, __labels__ = createDataSet()


def main():
    # dataSet, labels = createDataSet()
    global dataSet, labels
    dataSet_tmp = __dataSet__[:]
    labels_tmp = __labels__[:]
    desicionTree = C45Tree(dataSet_tmp, labels_tmp)
    print('desicionTree:\n', desicionTree)
    treePlotter.createPlot(desicionTree)
    testSet = createTestSet()
    # print('predict result:\n', predict(desicionTree, labels, testSet))


if __name__ == '__main__':
    main()
