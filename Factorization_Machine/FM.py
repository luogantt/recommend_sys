

# coding:UTF-8

from __future__ import division
from math import exp
from numpy import *
from random import normalvariate  # 正态分布
from datetime import datetime
import pandas as pd
import numpy as np
from math import log10
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
            
            xx0=dataMatrix[x]
            xx=np.array(xx0)
            xx1=xx.T@xx
            vv=v@v.T
            e=xx1*vv
            interaction=0.5*(e.sum()-e.trace())
            p = w_0 + xx@w + interaction
            loss = (1-sigmoid(classLabels[x] * p[0, 0]) )   #计算损失
            w_0 = w_0 +alpha * loss * classLabels[x]

            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] +alpha * loss * classLabels[x] * xx[0,i]#dataMatrix[x, i]
                    for j in range(k):
                        vv=np.array([v[:,j]])
                        v[i, j] = v[i, j]+ alpha * loss * classLabels[x] * xx[0,i]*( xx@vv.T - v[i, j] * xx[0,i])
        print("第{}次迭代后的损失为{}".format(it, loss))

    return w_0, w, v


def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):   #计算每一个样本的误差
        allItem += 1
        xx0=dataMatrix[x]
        xx=np.array(xx0)
        xx1=xx.T@xx
        vv=v@v.T
        e=xx1*vv
        interaction=0.5*(e.sum()-e.trace())
        p = w_0 + xx@w + interaction

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
    w_0, w, v = SGD_FM(mat(dataTrain), labelTrain, 20, 30)
    print(
        "训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print(
    "训练用时为：%s" % (date_endTrain - date_startTrain))
    print("开始测试")
    print(
        "测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v)))
