#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:50:43 2020

@author: ledi
"""



import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from feature_column import build_input_features, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from layers.core import PredictionLayer, DNN
from layers.interaction import FM,CIN
from layers.utils import concat_func, add_func, combined_dnn_input
# from  deepfm import DeepFM

from keras.layers import Dense
# from feature_column import SparseFeat, DenseFeat, get_feature_names

# if __name__ == "__main__":
data = pd.read_csv('./criteo_sample.txt')


#离散的特征名称
sparse_features = ['C' + str(i) for i in range(1, 27)]

#数值的特征名称
dense_features = ['I' + str(i) for i in range(1, 14)]

#对缺失的特征进行填充
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )
target = ['label']


#数据预处理
# 1.Label Encoding for sparse features,and do simple Transformation for dense features
#对离散特征进行编码
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
#数值特征进行最大最小归一化
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])



#feature 是特征处理模块
from feature import Operate_Feat1,get_feature_names


d=Operate_Feat1()



sparse_list=[]
for p in sparse_features:
    d1=d.operate_sparse(data[p], p)
    sparse_list.append(d1.copy())

dense_list=[]
for q in dense_features:
    d2=d.operate_dense(q)
    print(d2)
    dense_list.append(d2.copy())


# fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
#                         for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
#                       for feat in dense_features]

merge_list=sparse_list+dense_list
dnn_feature_columns = merge_list
linear_feature_columns = merge_list








from feature import DEFAULT_GROUP_NAME,build_input_features

def xDeepFM(linear_feature_columns, dnn_feature_columns, dnn_hidden_units=(256, 256),
            cin_layer_size=(128, 128,), cin_split_half=True, cin_activation='relu', l2_reg_linear=0.00001,
            l2_reg_embedding=0.00001, l2_reg_dnn=0, l2_reg_cin=0, seed=1024, dnn_dropout=0,
            dnn_activation='relu', dnn_use_bn=False, task='binary'):

    
    
    # dnn_hidden_units=(256, 256)
    # cin_layer_size=(128, 128,)
    # cin_split_half=True
    # cin_activation='relu'
    # l2_reg_linear=0.00001
    # l2_reg_embedding=0.00001
    # l2_reg_dnn=0
    # l2_reg_cin=0
    # seed=1024
    # dnn_dropout=0
    # dnn_activation='relu'
    # dnn_use_bn=False
    # task='binary'

    features = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    inputs_list = list(features.values())
    from feature import get_linear_logit
    linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
                                    l2_reg=l2_reg_linear)
    from feature import input_from_feature_columns
    sparse_embedding_list, dense_value_list = input_from_feature_columns(features, dnn_feature_columns,
                                                                         l2_reg_embedding, seed)

    fm_input = concat_func(sparse_embedding_list, axis=1)

    dnn_input = combined_dnn_input(sparse_embedding_list, dense_value_list)
    dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed)(dnn_input)
    
        
    import keras 
    import tensorflow as tf
    dnn_logit = tf.keras.layers.Dense(
        1, use_bias=False, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(dnn_output)

    final_logit = add_func([linear_logit, dnn_logit])

    if len(cin_layer_size) > 0:
        exFM_out = CIN(cin_layer_size, cin_activation,
                       cin_split_half, l2_reg_cin, seed)(fm_input)
        exFM_logit = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.glorot_normal(seed))(exFM_out)
        final_logit = add_func([final_logit, exFM_logit])

    output = PredictionLayer(task)(final_logit)

    model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
    return model


 

model = xDeepFM(linear_feature_columns, dnn_feature_columns, task='binary')

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

# 3.generate input data for model

train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
# model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=2, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
