# -*- coding:utf-8 -*-

from __future__ import division, print_function

import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from algorithm.estimator import IterationEstimator


class ImplicitALS(IterationEstimator):
    """
    隐式交替最小二乘，果然不适合显式数据，表现很离谱
    
    属性
    ---------
    n_factors : 隐式因子数
    n_epochs : 迭代次数
    reg : 正则因子
    alpha : 隐式数据评分系数
    """

    def __init__(self, n_factors=20, n_epochs=10, reg=0.1, alpha=40):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.reg = reg
        self.alpha = alpha

    def alternative(self, X, Y, is_user):
        reg_I = self.reg * sparse.eye(self.n_factors)
        YTY = Y.T.dot(Y)
        I = sparse.eye(Y.shape[0])

        uids = self.train_dataset.uids if is_user else self.train_dataset.iids
        for u in uids:
            if is_user:
                ru = self.train_dataset.matrix.A[u]
            else:
                ru = self.train_dataset.matrix.A[:, u].T

            CuI = sparse.diags(ru * self.alpha, 0)
            Cu = CuI + I

            pu = ru.copy()
            pu[ru != 0] = 1.0

            YT_CuI_Y = Y.T.dot(CuI).dot(Y)
            YT_CuI_pu = Y.T.dot(Cu).dot(sparse.csr_matrix(pu).T)

            X[u] = spsolve(YTY + YT_CuI_Y + reg_I, YT_CuI_pu)

    def _prepare(self):
        self.user_num = self.train_dataset.matrix.shape[0]
        self.item_num = self.train_dataset.matrix.shape[1]
        self.X = sparse.csr_matrix(np.random.normal(size=(self.user_num, self.n_factors)))
        self.Y = sparse.csr_matrix(np.random.normal(size=(self.item_num, self.n_factors)))

    def _iteration(self):
        self.alternative(self.X, self.Y, True)
        self.alternative(self.Y, self.X, False)

    def _pred(self):
        return np.dot(self.X, self.Y.T)

    def predict(self, u, i):
        est = self.X[u].dot(self.Y[i].T)[0,0]
        return est