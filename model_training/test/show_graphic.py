# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:graphic.py

@time:2017/5/29 12:00

@desc:

"""
import numpy as np
from pykalman import KalmanFilter
import pandas as pd
import json
import os
from matplotlib import pyplot as plt

def Kalman(dataset):
    mean = np.mean(dataset)
    covariance = np.cov(dataset)
    kf = KalmanFilter(n_dim_obs=1,n_dim_state=1,initial_state_mean=mean,initial_state_covariance=covariance,
                      transition_matrices=[1],observation_matrices=[1],observation_covariance=covariance,
                      transition_covariance=np.eye(1),transition_offsets=None)
    filter = kf.filter(dataset)[0]
    return np.reshape(filter,(1,-1))[0]

def load_json_file(filename):
    """
    decoding json文件
    :param filename: file path
    :return: dict或list
    """
    json_file = open(filename,'r',encoding='utf-8')
    return json.load(json_file)


#load dataset

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'testoutput'))
BP_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"BP_NO"))
BP_YES =load_json_file("%s/history_0116_%s.json"%(output_path,"BP_YES"))
RNN_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"RNN_NO"))
RNN_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"RNN_YES"))
FBP_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"FBP_NO"))
FBP_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"FBP_YES"))
FRNN_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"FRNN_NO"))
FRNN_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"FRNN_YES"))
#
# nlist = []
# nlist.append(BP_NO)
# nlist.append(BP_YES)
# nlist.append(RNN_NO)
# nlist.append(RNN_YES)
# nlist.append(FBP_NO)
# nlist.append(FBP_YES)
# nlist.append(FRNN_NO)
# nlist.append(FRNN_YES)
# panda = pd.DataFrame({
# 'BP_NO':BP_NO['test']['mean_absolute_error'],
# 'BP_YES':BP_YES['test']['mean_absolute_error'],
# 'RNN_NO':RNN_NO['test']['mean_absolute_error'],
# 'RNN_YES':RNN_YES['test']['mean_absolute_error'],
# 'FBP_NO':FBP_NO['test']['mean_absolute_error'],
# 'FBP_YES':FBP_YES['test']['mean_absolute_error'],
# 'FRNN_NO':FRNN_NO['test']['mean_absolute_error'],
# 'FRNN_YES':FRNN_YES['test']['mean_absolute_error']
# },index=['test'])

width = 0.3

index = np.array([1,2])
feature1 = [RNN_NO['test']['mean_absolute_error'],RNN_YES['test']['mean_absolute_error']]
feature2 = [FRNN_NO['test']['mean_absolute_error'],FRNN_YES['test']['mean_absolute_error']]
print(feature1)
rect1 = plt.bar(index,feature1,width,color='b',label='time series')
rect2 = plt.bar(index+width,feature2,width,color='r',label='complex')
plt.xticks((1+width,2+width),('no kalman','kalman'),rotation=0)
plt.xlabel('model name')
plt.ylabel('Test mean_absolute_error')
plt.title('Compare features(NN:RNN)')

plt.legend(loc='upper center')
plt.show()
