#coding=utf-8
__author__ = 'luchi.lc'
import numpy as np

"""
date:29/6/2017
usage:构造GBDT树并用其生成数据新的特征向量
"""
class GBDT(object):

    def __init__(self,config):
        
        

        self.learningRate = config.learningRate #learning_rate
        self.maxTreeLength=config.maxTreeLength #树的最大深度
        self.maxLeafCount=config.maxLeafCount #最大叶子数量　
        self.maxTreeNum=config.maxTreeNum      #树的数量
        self.tree=[]

    #计算平方损失
    def calculateSquareLoss(self,residual):
        """
        :param residual:梯度残差值
        :return:总体的残差值
        """

        #如果这批数据的残差相同，那么loss为0
        mean = np.mean(residual)
        sumError = np.sum([(value-mean)**2 for value in residual])
        return sumError

    def splitTree(self,x_train,residualGradient,treeHeight):
        """

        :param x_train:训练数据
        :param residualGradient:当前需要拟合的梯度残差值
        :param treeHeight:树的高度
        :return:建好的GBDT树
        """
        size = len(x_train) #数据的数量
        dim = len(x_train[0]) #特征的维度
        #约定：左子树是小于等于，右子树是大于
        bestSplitPointDim=-1
        bestSplitPointValue=-1
        #这是树分裂前，loss
        curLoss = self.calculateSquareLoss(residualGradient)
        minLossValue=curLoss
        #如果树的递归深度等于树的最大深度，则递归终止
        if treeHeight==self.maxTreeLength:

            return curLoss
        tree=dict([])
        #遍历数据所有的维度
        for i in range(dim):
            #遍历所有的数据
            for j in range(size):
                #令　x_train[j,i]为分裂点    
                splitNum = x_train[j,i]
                leftSubTree=[]
                rightSubTree=[]
                #以splitNum 为分裂点，对于第i个feature ,将数据分成两类，
                for k in range(size):
                    tmpNum=x_train[k,i]
                    if tmpNum<=splitNum:
                        leftSubTree.append(residualGradient[k])
                    else:
                        rightSubTree.append(residualGradient[k])
                sumLoss=0.0
                #分别计算左右子树的loss,再求和，通过最小化loss,来决定分裂的feature和分裂的值
                sumLoss+=self.calculateSquareLoss(np.array(leftSubTree))
                sumLoss+=self.calculateSquareLoss(np.array(rightSubTree))
                if sumLoss<minLossValue:
                    
                    bestSplitPointDim=i
                    bestSplitPointValue=splitNum
                    minLossValue=sumLoss
                    print(treeHeight,bestSplitPointDim)
        '''
        通过上面的多轮循环，这个树某个节点的最优分裂特征和分裂值
        '''
        #如果损失值没有变小，则不作任何改变，也就是下面的归位一个Node
        if minLossValue==curLoss:
            return np.mean(residualGradient)
        else:
            #上面已经找到节点的特征和分裂值，那就用这个特征bestSplitPointDim，和值bestSplitPointValue将数据分叉，分裂成一个二叉树
            leftSplit=[(x_train[i],residualGradient[i]) for i in range(size) if x_train[i,bestSplitPointDim]<=bestSplitPointValue ]#左子树
            rightSplit=[(x_train[i],residualGradient[i]) for i in range(size) if x_train[i,bestSplitPointDim]>bestSplitPointValue ]#右子树
             
#            print(leftSplit)
            newLeftSubTree = list(zip(*leftSplit))[0] #左子树的训练数据X
            newLeftResidual = list(zip(*leftSplit))[1]#左子树的y
            leftTree = self.splitTree(np.array(newLeftSubTree),newLeftResidual,treeHeight+1)

            newRightSubTree = list(zip(*rightSplit))[0]
            newRightResidual =list(zip(*rightSplit))[1]
            rightTree = self.splitTree(np.array(newRightSubTree),newRightResidual,treeHeight+1)

            tree[(bestSplitPointDim,bestSplitPointValue)]=[leftTree,rightTree]
            
            print(tree)
            return tree

    #计算树的节点数
    def getTreeLeafNodeNum(self,tree):
            size=0
            if type(tree) is not dict:
                return 1
            for item in tree.items():
                
                print(item)
                
                print('#'*10)
                subLeftTree,subRightTree=item[1]
                if type(subLeftTree) is dict:
                    size+=self.getTreeLeafNodeNum(subLeftTree)
                else:
                    size+=1

                if type(subRightTree) is dict:
                    size+=self.getTreeLeafNodeNum(subRightTree)
                else:
                    size+=1
            return size

    #遍历数据应该归到那个叶子节点，并计算其左侧的叶子节点个数
    def scanTree(self,curTree,singleX,treeLeafNodeNum):
        """

        :param curTree:当前的树
        :param singleX:需要送入到决策树的数据
        :param treeLeafNodeNum:树的叶子结点个数
        :return:该数据应该分到的叶子结点的值和其在当前树的转换的特征向量
        """

        self.xValue=0
        xFeature=[0]*treeLeafNodeNum
        self.leftZeroNum=0
        def scan(curTree,singleX):

            for item in curTree.items():
                splitDim,splitValue=item[0]
                subLeftTree,subRightTree=item[1]
                if singleX[splitDim]<=splitValue:
                    if type(subLeftTree) is dict:
                        scan(subLeftTree,singleX)
                    else:
                        self.xValue=subLeftTree
                        return
                else:
                    self.leftZeroNum+=self.getTreeLeafNodeNum(subLeftTree)
                    if type(subRightTree) is dict:
                        scan(subRightTree,singleX)
                    else:
                        self.xValue=subRightTree
                        return
        scan(curTree,singleX)
        xFeature[self.leftZeroNum]=1
        return self.xValue,xFeature

    #sigmoid函数
    def sigmoid(self,x):
        return 1.0/(1+np.exp(-1*x))
    #建立GBDT树
    def buildGbdt(self,x_train,y_train):
        #数据的个数
        size = len(x_train)
        dim = len(x_train[0])
        x_train=np.array(x_train)
        y_train=np.array(y_train)
        x_train_feature=[]

        #初始化第一棵树
        treePreviousValue=0*y_train
        treeValues=[]
        treeValues.append(treePreviousValue)

        curValue = self.sigmoid(0*y_train)
        dataFeatures=[]
        for i in range(self.maxTreeNum):
            print("the tree %i-th"%i)
            residualGradient = -1*self.learningRate*(curValue-y_train)
            curTree = self.splitTree(x_train,residualGradient,1)
            self.tree.append(curTree)
#            print (curTree)
            #更新梯度残差值
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            curTreeValue=[]
            for singleX in x_train:
                xValue,xFeature = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curTreeValue.append(xValue)

            treePreviousValue=np.array(curTreeValue)+treePreviousValue
            curValue=self.sigmoid(treePreviousValue)
#            print (y_train)
#            print("curValue")
#            print( curValue)

    #根据建成的树构建输入数据的特征向量
    def generateFeatures(self,x_train):
        dataFeatures=[]
        for curTree in self.tree:
            curFeatures=[]
            curTreeLeafNodeNum = self.getTreeLeafNodeNum(curTree)
            # print ("tree leaf node is %i"%(curTreeLeafNodeNum))
            for singleX in x_train:
                _,xFeature = self.scanTree(curTree,singleX,curTreeLeafNodeNum)
                curFeatures.append(xFeature)

            if len(dataFeatures)==0:
                dataFeatures=np.array(curFeatures)
            
            else:
                dataFeatures=np.concatenate([dataFeatures,curFeatures],axis=1)
                
#                print('#'*100)
#                print(len(curFeatures[0]),len(dataFeatures[0]))
#        print('data_feature=',dataFeatures,len(dataFeatures),len(dataFeatures[0]))
#        print('curFeatures=',curFeatures)
        return dataFeatures








