# -*- coding: utf-8 -*-
# @Time    : 2018/8/15 10:27
# @Author  : Lemon_shark
# @Email   : jiping_cehn@163.com
# @File    : FM.py
# @Software: PyCharm Community Edition

# coding:UTF-8

from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np
import math
trainData = 'data/diabetes_train.txt'   #请换为自己文件的路径
testData = 'data/diabetes_test.txt'

def preprocessData(data):
    feature=np.array(data.iloc[:,:-1])   #取特征
    label=data.iloc[:,-1].map(lambda x: 1 if x==1 else -1) #取标签并转化为 +1，-1
    #将数组按行进行归一化
    zmax, zmin = feature.max(axis=0), feature.min(axis=0)
    feature = (feature - zmin) / (zmax - zmin)
    label=np.array(label)

    return feature,label

def sigmoid(inx):
    #return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return 1.0 / (1 + exp(-inx))

def SGD_FM(dataMatrix, classLabels, k, iter):
    '''
    :param dataMatrix:  特征矩阵
    :param classLabels: 类别矩阵
    :param k:           辅助向量的大小
    :param iter:        迭代次数
    :return:
    '''
    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)   #矩阵的行列数，即样本数和特征数
    alpha = 0.01
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))      #一阶特征的系数
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))   #即生成辅助向量，用来训练二阶交叉特征的系数

    for it in range(iter):
        for x in range(m):  # 随机优化，每次只使用一个样本
            # 二阶项的计算
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)  #二阶交叉项的计算
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.       #二阶交叉项计算完成

            p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出，即FM的全部项之和
#            loss = 1-sigmoid(classLabels[x] * p[0, 0])    #计算损失
            loss =-math.log10(sigmoid(classLabels[x] * p[0, 0])  )  #计算损失
            w_0 = w_0 +alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * (
                        dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])
        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    return float(error) / allItem


if __name__ == '__main__':
    train=pd.read_csv(trainData)
    test = pd.read_csv(testData)
    dataTrain, labelTrain = preprocessData(train)
    dataTest, labelTest = preprocessData(test)
    date_startTrain = datetime.now()
    print    ("开始训练")
    w_0, w, v = SGD_FM(mat(dataTrain), labelTrain, 20, 200)
    print(
        "训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print(
    "训练用时为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print(
        "测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))
