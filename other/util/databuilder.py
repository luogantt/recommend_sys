# -*- coding:utf-8 -*-

import itertools
import os

import numpy as np
from scipy.sparse import csr_matrix
from util.matrix import Matrix
import util.tools as tl


class DataBuilder(object):
    """
    构造数据模型
       
    参数
    ----------    
    file_name : 文件地址，这里用的grouplens数据集
    shuffle : 是否对数据shuffle
    just_test_one : k折交叉验证要运行k次，这里只运行一次，方便测试程序正确性
    """

    def __init__(self, file_name, shuffle=True, just_test_one=False):
        self.file_name = file_name
        self.shuffle = shuffle
        self.just_test_one = just_test_one

    def read_ratings(self):
        """
        读取数据
        """

        with open(os.path.expanduser(self.file_name)) as f:
            raw_ratings = [self.parse_line(line) for line in itertools.islice(f, 0, None)]
        return raw_ratings

    def parse_line(self, line):
        line = line.split("\t")
        uid, iid, r, timestamp = (line[i].strip() for i in range(4))
        return uid, iid, float(r), timestamp

    def cv(self, k_folds):
        raw_ratings = self.read_ratings()

        if self.shuffle:
            np.random.shuffle(raw_ratings)

        stop = 0
        raw_len = len(raw_ratings)
        offset = raw_len // k_folds
        left = raw_len % k_folds
        for fold_i in range(k_folds):
            print("current fold {}".format(fold_i + 1))
            start = stop
            stop += offset
            if fold_i < left:
                stop += 1

            #使用生成器，提高效率
            yield self.mapping(raw_ratings[:start] + raw_ratings[stop:]), raw_ratings[start:stop]

    def mapping(self, raw_train_ratings):
        uid_dict = {}
        iid_dict = {}
        current_u_index = 0
        current_i_index = 0

        row = []
        col = []
        data = []
        for urid, irid, r, timestamp in raw_train_ratings:
            try:
                uid = uid_dict[urid]
            except KeyError:
                uid = current_u_index
                uid_dict[urid] = current_u_index
                current_u_index += 1
            try:
                iid = iid_dict[irid]
            except KeyError:
                iid = current_i_index
                iid_dict[irid] = current_i_index
                current_i_index += 1

            row.append(uid)
            col.append(iid)
            data.append(r)

        sparse_matrix = csr_matrix((data, (row, col)))

        return Matrix(sparse_matrix, uid_dict, iid_dict)

    def eval(self, algorithm, measures=["rmse", "mae"], k_folds=5):
        eval_results = []

        for train_dataset, test_dataset in self.cv(k_folds):
            algorithm.train(train_dataset)
            eval_results.append(algorithm.estimate(test_dataset, measures))
            if self.just_test_one:
                break

        tl.print_pretty(measures, eval_results)