# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:RNN_para.py

@time:2017/5/28 23:04

@desc:
绘制 参数比较 图表 ：激活函数和优化器
"""

import numpy
import os
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import json
from keras import backend as K
from keras.utils import plot_model

def creat_dataset(dataset,look_back = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return numpy.array(dataX),numpy.array(dataY)




#load dataset
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
data_matrix_files_name = source_path + '/' + "matrix"
data_matrix_file_name = "%s/%s_%s.csv" %(data_matrix_files_name, "火车东站", "0116")


dataframe = pandas.read_csv(data_matrix_file_name,encoding='utf-8',header=0,index_col=0)
dataset  = dataframe.iloc[:,-2].values



print(dataset)

# normalize

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# split into train and test
train_size = int(len(dataset)*0.8)
test_size = len(dataset)-train_size
look_back = 7 # maybe is 24h
train ,test= dataset[0:train_size] , dataset[train_size-look_back:len(dataset)]



# reshape
trainX,trainY = creat_dataset(train,look_back)
testX,testY = creat_dataset(test,look_back)
opt = ['adam','sgd','adagrad']
act = ['relu','sigmoid','tanh']
mae_list = {}
i = 0
# reshape input to be [sample,time steps,features]
# trainX = numpy.reshape(trainX,(trainX.shape[0],look_back,1))
# testX = numpy.reshape(testX,(testX.shape[0],look_back,1))
#
# # # reshape input to be [sample,time steps,features]
trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX = numpy.reshape(testX,(testX.shape[0],trainX.shape[1],1))
for o in opt:
    for activation in act:
        K.clear_session()

        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(5,input_dim= 1,activation=activation))
        model.add(Dense(1,activation='sigmoid'))
        model.compile(loss='mae',optimizer=o,metrics=['mae'])
        history = model.fit(trainX,trainY,nb_epoch=30,batch_size=1,verbose=2)
        cost = model.evaluate(testX,testY,batch_size=1,verbose=2)

        #
        # # reshape input to be [sample,time steps,features]
        # trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1]))
        # testX = numpy.reshape(testX,(testX.shape[0],testX.shape[1]))
        #
        # model = Sequential()
        # model.add(Dense(5,input_dim=7,activation=activation))
        # model.add(Dense(3,activation=activation))
        # model.add(Dense(2,activation=activation))
        # model.add(Dense(1))  # 输出层
        #
        # #     参数设置
        # model.compile(loss='mae', optimizer=o, metrics=['mae'])  # 编译模型必须的两个参数
        #
        # # 训练模型
        # history = model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=2)
        # # 测试模型
        # cost = model.evaluate(testX, testY, batch_size=1, verbose=2)

        # inverse
        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)
        mae_list[i] ={'name':o+'_'+activation,'history':history.history['mean_absolute_error']}
        i+=1


plt.figure(1)#创建图表1
plt.xlabel('epoch')
plt.ylabel('mean_absolute_error')
plt.title('parameters comparing in RNN Network')
# shift train predictions for plotting
style = ['o-','^-','s-']
color = ['g','r','b']
for i in range(3):
    for j in range(3):
        plt.plot(mae_list[i*3+j]['history'],style[i]+color[j],label=mae_list[i*3+j]['name'])
plt.legend()
plt.show()
