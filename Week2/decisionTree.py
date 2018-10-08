# -*- coding: utf-8 -*-
"""
decisionTree.py

决策树算法

Created on Mon Oct  8 16:40:16 2018

@author: Minxi Yang
"""

import numpy as np
import operator

from math import log

def calcShannonEnt(dataSet):
    """
    计算香农熵
    参数：
        dataSet -- 数据集，包含不同类型
    输出：
        shannonEnt -- 香农熵
    """
    #求数据集大小
    numEntries = len(dataSet)
    #新建一个用于统计各个类型数目的字典
    labelCounts = {}
    #遍历数据集中的每个数据
    for featVec in dataSet:
        #类型为数据中的最后一项
        currentLabel = featVec[-1]
        #如果之前没有记录该类型
        if currentLabel not in labelCounts.keys():
            #添加新的一项记录
            labelCounts[currentLabel] = 0
        #计数
        labelCounts[currentLabel] += 1
    #初始化香农熵
    shannonEnt = 0.0
    #遍历所有类型
    for key in labelCounts:
        #计算概率，古典概型
        prob = np.float(labelCounts[key])/numEntries
        #根据公式计算香农熵
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#创造点数据
def createDataSet():
    """
    创造测试数据
    参数：
        无
    输出：
        dataSet -- 数据集
        labels -- 标签
    """
    dataSet = [[1, True, 'yes'],
              [1, True, 'yes'],
              [1, False, 'no'],
              [0, True, 'no'],
              [0, True, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    """
    按给定特征划分数据集
    参数：
        dataSet -- 待划分的数据集
        axis -- 划分的参照特征
        value -- 分界线的特征值
    """
    #新建存放划分后数据集的列表
    retDataSet = []
    #遍历数据集中的书籍
    for featVec in dataSet:
        #找到划分界限
        if featVec[axis] == value:
            #取之前的特征，不包括参照特征
            reducedFeatVec = featVec[:axis]
            #取之后的特征，不包括参照特征
            reducedFeatVec.extend(featVec[axis+1:])
            #添加到划分数据集中
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    选择最好的划分参照特征，也就是使得信息增益最大的划分方式
    参数：
        dataSet -- 数据集
    返回：
        bestFeature -- 最好的数据集
    """
    #求特征数，-1是去掉标签
    numFeatures = len(dataSet[0]) - 1
    #求数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    #初始化最佳信息增益，和最好的参照特征
    bestInfoGain = 0.0
    bestFeature = -1
    #遍历每个特征
    for i in range(numFeatures):
        #取每个数据的第i个特征值
        featList = [example[i] for example in dataSet]
        #把这些特征值作为一个集合
        uniqueVals = set(featList)
        #初始化划分后的熵
        newEntropy = 0.0
        #遍历集合中的特征值
        for value in uniqueVals:
            #取第i个特征值为value子集
            subDataSet = splitDataSet(dataSet, i, value)
            #计算概率
            prob = len(subDataSet) / np.float(len(dataSet))
            #计算划分后的熵，为各个子集熵的期望
            newEntropy += prob * calcShannonEnt(subDataSet)
        #计算信息增益
        infoGain = baseEntropy - newEntropy
        #如果当前以i特征划分得到的信息增益大于之前的最大值
        if (infoGain > bestInfoGain):
            #加冕为王
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    """
    求出现次数最多分类的名称
    参数：
        classList -- 类列表
    返回：
        sortedClassCount[0][0] -- 最多分类的名称
    """
    #新建一个类计数器，字典
    classCount = {}
    #遍历类列表
    for vote in classList:
        #如果第一次遇到一个类
        if vote not in classCount.keys():
            #新建一个类计数器
            classCount[vote] = 0
        #计数+1
        classCount[vote] += 1
        #排序
    sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse = True)
    return sortedClassCount[0][0]

def createTree(dataSet, featLabels):
    """
    创建决策树的递归函数
    参数：
        dataSet -- 数据集
        labels -- 标签
    返回：
        myTree -- 创建的决策树
    """
    #复制一份特征列表，因为后边要对labels做del，会改变原参数
    labels = list.copy(featLabels)
    #得到类列表
    classList = [example[-1] for example in dataSet]
    #如果类列表中只有一类
    if classList.count(classList[0]) == len(classList):
        #返回该类，结束函数
        return classList[0]
    #如果数据只有类，没有数据，也就是用尽所有特征也没法把数据划分成单一类
    if len(dataSet[0]) == 1:
        #返回最多的那一类，结束函数
        return majorityCnt(classList)
    #最佳分类特征的序号
    bestFeat = chooseBestFeatureToSplit(dataSet)
    #最佳分类标签
    bestFeatLabel = labels[bestFeat]
    #初始化决策树，字典型
    myTree = {bestFeatLabel:{}}
    #从列表中删除划分过的类
    del(labels[bestFeat])
    #得到特征值
    featValues = [example[bestFeat] for example in dataSet]
    #值的集合
    uniqueVals = set(featValues)
    #遍历所有值
    for value in uniqueVals:
        #剩余的类别名称
        subLabels = labels[:]
        #调用子节点递归函数
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    #返回给父节点迭代函数
    return myTree

def classify(inputTree, featLabels, testVec):
    """
    使用决策树执行分类，递归函数
    参数：
        inputTree -- 输入决策树，字典
        featLabels -- 特征名的列表
        testVec -- 测试向量
    返回：
        classLabel -- 预测分类
    """
    #根节点类型
    firstStr = list(inputTree.keys())[0]
    #根节点对应字典
    secondDict = inputTree[firstStr]
    #求根特征对应的特征序号
    featIndex = featLabels.index(firstStr)
    #遍历所有关键词
    for key in secondDict.keys():
        #匹配到了待分类向量的第featIndex个特征
        if testVec[featIndex] == key:
            #不是叶子节点，节点元素是一个字典
            if type(secondDict[key]).__name__ == 'dict':
                #对这个字典递归调用分类函数
                classLabel = classify(secondDict[key],featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(inputTree, filename):
    """
    存储决策树
    参数：
        inputTree -- 输入决策树
        filename -- 文件名
    返回：
        无 -- 直接存储文件
    """
    import pickle
    #可写的文件指针
    fw = open(filename, 'wb')
    #存储决策树
    pickle.dump(str(inputTree), fw)
    #关闭指针
    fw.close()

def grabTree(filename):
    """
    提取决策树
    参数：
        filename -- 文件名
    返回：
        tree -- 读取到的决策树
    """
    import pickle
    #打开只读文件指针
    fr = open(filename, 'rb')
    #载入文件内容
    tree = pickle.load(fr)
    return tree

