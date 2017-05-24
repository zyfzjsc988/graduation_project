# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:train_result.py

@time:2017/5/22 19:41

@desc:
根据模型在线测试
"""
import pandas as pd
import numpy as np
import json
from keras.models import model_from_json

def separate_train_and_test(target_matrix,tag_list):
    """
    分离测试集和训练集
    :param target_matrix: 目标集
    :param tag_list: 标签集
    :return: 
    """

    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    for i in range(len(tag_list)):
        if tag_list[i] == 1 or tag_list[i] == "1":
            X_test.append(target_matrix[i,:-1])
            Y_test.append(target_matrix[i,-1])
        else:
            X_train.append(target_matrix[i,:-1])
            Y_train.append(target_matrix[i, -1])


    return np.array(X_train),np.array(Y_train),np.array(X_test),np.array(Y_test)

class Predict(object):
    matrix_path = ""
    history_path = ""
    model_path =""
    weights_path = ""
    def __init__(self,matrix_path,history_path,model_path,weights_path):
        self.matrix_path = matrix_path
        self.history_path = history_path
        self.model_path = model_path
        self.weights_path = weights_path

    def train_and_predict(self):
        print("-----2----")
        print(self.matrix_path)
        print(self.history_path)
        print(self.model_path)
        print(self.weights_path)
        history = json.load(open(self.history_path, 'r', encoding='utf-8'))
        print(history)
        return history
