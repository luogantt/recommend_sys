#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 19:49:32 2020

@author: ledi
"""
DEFAULT_GROUP_NAME = "default_group"
from collections import namedtuple
from tensorflow.python.keras.initializers import RandomNormal, Zeros
import pandas as pd
from collections import OrderedDict
from tensorflow.python.keras.layers import Input
from layers import Linear
from layers.utils import concat_func, add_func


class Operate_Feat1():
    def __init__(self):
        
        #这里是类别特征的内置配置
        self.sparse_dict={  'embedding_dim':4, 'use_hash':False,'dtype':"int32", 
            
            'feat_cat':'sparse',
            'embeddings_initializer':RandomNormal(mean=0.0, stddev=0.0001, seed=2020),
           
             'embedding_name':None,'group_name':"default_group", 'trainable':True}
        #这里是数值特征的内置配置
        self.dense_dict={'dimension':1, 'dtype':"float32", 'feat_cat':'dense',}
    
    
    #结果都以字典的形式输出
    def operate_sparse(self,some_data,name):
        sparse_dict1=self.sparse_dict
        sparse_dict1['vocabulary_size']=some_data.nunique()
        sparse_dict1['embedding_name'] =name
        return pd.Series(sparse_dict1)
    def operate_dense(self,dense_name):
        dense_dict1=self.dense_dict
        dense_dict1['name']=dense_name
        
        return pd.Series(dense_dict1)
    
# 构建输入层
def build_input_features(feature_columns, prefix=''):
    input_features = OrderedDict()
    for fc in feature_columns:
        if fc['feat_cat'] == 'sparse':
            input_features[fc['embedding_name']] = Input(
                shape=(1,), name=prefix + fc['embedding_name'], dtype=fc['dtype'])
        elif fc['feat_cat'] == 'dense':
            input_features[fc['name']] = Input(
                shape=(fc['dimension'],), name=prefix + fc['name'], dtype=fc['dtype'])


        else:
            raise TypeError("Invalid feature column type,got", type(fc))

    return input_features

def get_feature_names(feature_columns):
    features = build_input_features(feature_columns)
    
    print('features==============',features)
    return list(features.keys())


def get_linear_logit(features, linear_feature_columns, units=1, use_bias=False, seed=1024, prefix='linear',
                     l2_reg=0):
    
    features=features
    linear_feature_columns=linear_feature_columns
    units=1
    use_bias=False
    seed=1024
    prefix='linear'
    l2_reg=0
    
    

    for i in range(len(linear_feature_columns)):
        if linear_feature_columns[i]['feat_cat']=='sparse':
            linear_feature_columns[i]['embedding_dim']=3
            linear_feature_columns[i]['embeddings_initializer']=Zeros()
                                                                           


    linear_emb_list = [input_from_feature_columns(features, linear_feature_columns, l2_reg, seed,prefix=prefix + str(i))[0] for i in range(units)]
    _, dense_input_list = input_from_feature_columns(features, linear_feature_columns, l2_reg, seed, prefix=prefix)

    linear_logit_list = []
    for i in range(units):

        if len(linear_emb_list[i]) > 0 and len(dense_input_list) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=2, use_bias=use_bias, seed=seed)([sparse_input, dense_input])
        elif len(linear_emb_list[i]) > 0:
            sparse_input = concat_func(linear_emb_list[i])
            linear_logit = Linear(l2_reg, mode=0, use_bias=use_bias, seed=seed)(sparse_input)
        elif len(dense_input_list) > 0:
            dense_input = concat_func(dense_input_list)
            linear_logit = Linear(l2_reg, mode=1, use_bias=use_bias, seed=seed)(dense_input)
        else:
            # raise NotImplementedError
            return add_func([])
        linear_logit_list.append(linear_logit)

    return concat_func(linear_logit_list)



def input_from_feature_columns(features, feature_columns, l2_reg, seed, prefix='', seq_mask_zero=True,
                               support_dense=True, support_group=False):
    # feature_columns=linear_feature_columns
    # seq_mask_zero=True
    # support_dense=True
    # support_group=False
    
    print('KKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKKK')
    
    print('prefix=',prefix)
    sparse_feature_columns=[]
    for fc in feature_columns:
        if  fc['feat_cat'] == 'sparse':
            print(fc['feat_cat'])
            sparse_feature_columns.append(fc)

    # varlen_sparse_feature_columns = list(
    #     filter(lambda x: isinstance(x, VarLenSparseFeat), feature_columns)) if feature_columns else []
    
    
    '''
    {'C1': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de6377910>,
     'C2': <tensorflow.python.keras.layers.embeddings.Embedding at 0x7f5de62dd1c0>}
    '''
    
    from inputs1 import create_embedding_dict,create_embedding_matrix,get_dense_input

    #embedding_matrix_dict是一个字典，key 是特征的名称，values 是某个特征的Embedding
    embedding_matrix_dict = create_embedding_matrix(feature_columns, l2_reg, seed, prefix=prefix,
                                                        seq_mask_zero=seq_mask_zero)
    from inputs1 import embedding_lookup
    #group_sparse_embedding_dict 是每个特征从input层到embedding 层的映射 ,
    #这是一个列表
    group_sparse_embedding_dict = embedding_lookup(embedding_matrix_dict, features, sparse_feature_columns)
    
    
    
    #获得dense的输入
    dense_value_list = get_dense_input(features, feature_columns)

    
    print('TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTT')
    return group_sparse_embedding_dict, dense_value_list
