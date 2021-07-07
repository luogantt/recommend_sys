# -*- coding:utf-8 -*-
"""

Author:
    Weichen Shen,wcshen1994@163.com

"""

from collections import defaultdict
from itertools import chain

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2

from layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from layers.utils import Hash



from keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2
def create_embedding_dict(sparse_feature_columns ,seed, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    
    #将特征进行embedding ,输入维度是某个特征的种类数
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    return sparse_embedding



def get_dense_input(features, feature_columns):
    # import feature_column as fc_lib
    dense_feature_columns=[]
    for fc in feature_columns:
        if  fc['feat_cat'] == 'dense':
            dense_feature_columns.append(fc)
    dense_input_list = []
    for fc in dense_feature_columns:
        dense_input_list.append(features[fc['name']])
    return dense_input_list


def create_embedding_matrix(feature_columns, l2_reg, seed, prefix="", seq_mask_zero=True):


    sparse_feature_columns=[]
    for fc in feature_columns:
        if  fc['feat_cat'] == 'sparse':
            sparse_feature_columns.append(fc)

    sparse_emb_dict = create_embedding_dict(sparse_feature_columns,  seed,
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, sparse_input_dict, sparse_feature_columns, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    
    # sparse_embedding_dict=embedding_matrix_dict
    # sparse_input_dict =features
    
     # =sparse_feature_columns
    group_embedding_dict = []
    for fc in sparse_feature_columns:
        feature_name = fc.embedding_name
        embedding_name = fc.embedding_name
        # if (len(return_feat_list) == 0 or feature_name in return_feat_list):
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(
                sparse_input_dict[feature_name])
        else:
            
            # 模型输入层张量
            lookup_idx = sparse_input_dict[feature_name]
                                                       # 从输入层到embedding 层的映射
        group_embedding_dict.append(sparse_embedding_dict[embedding_name](lookup_idx))

    return group_embedding_dict
#这里面是从input 到embedding 层的映射 
