import os

from algorithm.mf.baseline import Baseline
from util.databuilder import DataBuilder

from algorithm.dnn.neumf import NeuMF

from algorithm.mf.explicit_als import ExplicitALS
from algorithm.mf.svd import SVD
from algorithm.mf.svdpp import SVDPlusPlus
from algorithm.mf.implicit_als import ImplicitALS

from algorithm.neighborhood.slop_one import SlopOne
from algorithm.neighborhood.itemcf import Itemcf

file_name = os.path.abspath("data/ml-100k/u.data")
data_builder = DataBuilder(file_name, just_test_one=True)



data_builder.eval(NeuMF(epochs=2))



data_builder.eval(Itemcf())



data_builder.eval(SlopOne())



data_builder.eval(Baseline())


data_builder.eval(SVD())


data_builder.eval(SVDPlusPlus())


data_builder.eval(ExplicitALS())


data_builder.eval(ImplicitALS())
 
 