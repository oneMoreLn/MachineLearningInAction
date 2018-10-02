# -*- coding: utf-8 -*-
"""
KNN.py

Created on Mon Oct  1 17:44:16 2018

@author: Minxi Yang
"""
#%%
import numpy as np
import operator
import matplotlib
import matplotlib.pyplot as plt
#%%

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """
    k近邻算法函数，分为三部分：计算距离，选择最近的k个点，排序
    参数：
        inX -- 需要判断的输入向量
        dataSet -- 输入的训练样本
        labels -- 标签向量
        k -- 近邻点的个数
    """
    #（1）计算距离
    #求数据集大小，要求样例垂直排列
    dataSetSize = dataSet.shape[0]
    #求差矩阵，向量化，一次对所有训练样本求差
    #np.tile()相当于MATLAB的repmat，周期延拓矩阵
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #差矩阵平方
    sqDiffMat = np.power(diffMat, 2)
    #对平方求和，沿水平方向
    sqDistances = np.sum(sqDiffMat, axis = 1, keepdims = True)
    #再开方
    distances = np.sqrt(sqDistances)
    #(2)选择距离最小的k个点
    #排序用np.argsort()能返回升序排序后的索引
    sortedDistIndicies = np.argsort(distances, axis = 0)
    #类计数器，是一个Python字典
    classCount = {}
    #距离最小的点最近，优先找
    for i in range(k):
        #取得第i近的点的类型
        voteIlabel = labels[np.int(sortedDistIndicies[i])]
        #该类型计数+1
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    #按照键值从大到小排列类计数器
    sortedClassCount = np.array(sorted(classCount, key = lambda x:classCount[x], reverse = True)).tolist()
    return sortedClassCount[0]

#%%

group, labels = createDataSet()
print(classify0([0, 0], group, labels, 3))
#%%
def file2matrix(filename):
    """
    将文本记录转换为numpy
    参数：
        filename -- 文件名
    返回：
        returnMat -- 转换的矩阵
        classLabelVector -- 类标签向量
    """
    #打开文件
    fr = open(filename)
    #一行一行读
    arrayOLines = fr.readlines()
    #求行数
    numberOfLines = len(arrayOLines)
    #为返回矩阵和标签向量预分配空间
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    classifiedData = {'didntLike ':[], 'smallDoses':[], 'largeDoses':[]}
    index = 0
    for line in arrayOLines:
        #str.strip()去掉每行前后的空格
        line = line.strip()
        #按制表符'\t'分割
        listFromLine = line.split('\t')
        #把数据中的前三项（特征数据）放在数据矩阵对应行中
        returnMat[index, :] = listFromLine[0:3]
        #把数据中的倒数第一项（标签）添加到标签向量中
        classLabelVector.append(np.uint32(listFromLine[-1]))
        #把数据分类添加到分类数据集中
        classifiedData[listFromLine[-2]].append(np.array(listFromLine[0:3]))
        #序号+1，这里使用序号++的形式是因为执行循环的次数是由行数决定的
        index += 1
    classifiedData['didntLike '] = np.array(classifiedData['didntLike '])
    classifiedData['smallDoses'] = np.array(classifiedData['smallDoses'])
    classifiedData['largeDoses'] = np.array(classifiedData['largeDoses'])
    return returnMat, classLabelVector, classifiedData
#%%
datingDataMat, datingLabels, classifiedData = file2matrix('datingTestSet.txt')

#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(classifiedData['didntLike '][:, 0], classifiedData['didntLike '][:, 1], marker = 'o', color = 'y')
ax.scatter(classifiedData['smallDoses'][:, 0], classifiedData['smallDoses'][:, 1], marker = 'x', color = 'g')
ax.scatter(classifiedData['largeDoses'][:, 0], classifiedData['largeDoses'][:, 1], marker = '^', color = 'b')
plt.legend(("didn't like", "small doses", "largeDose"))
plt.title("Visualized dating data")
plt.xlabel("Annual flight distance / mile")
plt.ylabel("Video game time / %")
#%%
def autoNorm(dataSet):
    """
    归一化特征值
    归一化到区间[0, 1]
    参数：
        dataSet -- 数据集
    返回：
        normDataSet -- 归一化后的数据集
        range -- 极差
        minVals -- 最小值
    """
    #求数据集最大最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值-最小值=极差
    ranges = maxVals - minVals
    normDataSet = np.zeros_like(dataSet)
    #求数据集大小
    m = dataSet.shape[0]
    #(数据集-最小值)/极差，归一化
    #其实这里也可以不用np.tile()，用Python广播的性质，但是可读性如这个强
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals
#%%
normMat, ranges, minVals = autoNorm(datingDataMat)
print("归一化后的数据：\n{}".format(normMat))
print("极差 = {}".format(ranges))
print("最小值 = {}".format(minVals))
#%%
def datingClassTest():
    #测试比例
    hoRatio = 0.10
    #读取数据
    datingDataMat, datingLabels, _ = file2matrix('datingTestSet.txt')
    #归一化数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #得到数据集大小
    m = normMat.shape[0]
    #求得测试数
    numTestVecs = np.int(m*hoRatio)
    #错误数记录
    errorCount = 0.0
    #遍历测试集
    for i in range(numTestVecs):
        #得到分类结果
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m],3)
        #打印预测结果
        print("The classifier came back with: {}, the real answer is: {}".format(classifierResult, datingLabels[i]))
        #统计错误
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    #打印正确率
    print("The accuracy is: {}".format(1 - errorCount/np.float(numTestVecs)))
#%%
datingClassTest()
#%%
def classifyPerson():
    """
    从键盘得到一组数据，并对判断其分类
    参数:
        None -- 直接从键盘读数据
    返回:
        None -- 直接打印结果
    """
    #结果列表
    resultList = ['not at all', 'in small doses', 'in large doses']
    #读取3个数据
    percentTats = np.float(input("Percentage of time spent playing video games =\n"))
    ffMiles = np.float(input("Frequent flier miles earned per year =\n"))
    iceCream = np.float(input("Liters of ice cream consumed per year =\n"))
    #读取样本数据
    datingDataMat, datingLabels, _= file2matrix('datingTestSet.txt')
    #标准化
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #数据合并为矩阵
    inArr = np.array([ffMiles, percentTats, iceCream])
    #预测
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    #打印结果
    print("You will probably like this person: {}".format(resultList[classifierResult - 1]))

#%%
classifyPerson()