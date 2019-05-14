__author__ = 'luchi.lc'

"""
date:29/6/2017
usage:测试GBDT的树的个数对结果的影响
"""
from GBDTReg import GBDT
class Config(object):
    learningRate=0.1
    maxTreeLength=4
    maxLeafCount=30
    maxTreeNum=50

def test():
    x=[[0.5,0.6,0.7],[0.4,0.5,0.5],[1.2,1.3,1.0],[1.4,1.5,0.8],[1.5,1.3,1.3]]
    y=[0,0,1,1,1]
    c=Config()
    gbdt=GBDT(config=c)
    gbdt.buildGbdt(x,y)
    data_features=gbdt.generateFeatures(x)
    print len(data_features[0])

test()
