# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:F_NN_test.py

@time:2017/5/27 9:51

@desc:交叉实验文件；用于更改测试集进行交叉实验

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


def creat_dataset(dataset,look_back,feature):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):
        f = feature[i+look_back]
        d = dataset[i:(i+look_back)]
        dataX.append(numpy.append(f,d))
        dataY.append(dataset[i+look_back])

    return numpy.array(dataX),numpy.array(dataY)

def creat_dataset1(dataset,look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)

def Kalman(dataset):
    mean = numpy.mean(dataset)
    covariance = numpy.cov(dataset)
    kf = KalmanFilter(n_dim_obs=1,n_dim_state=1,initial_state_mean=mean,initial_state_covariance=covariance,
                      transition_matrices=[1],observation_matrices=[1],observation_covariance=covariance,
                      transition_covariance=numpy.eye(1),transition_offsets=None)
    filter = kf.filter(dataset)[0]
    return numpy.reshape(filter,(1,-1))[0]



#load dataset
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
data_matrix_files_name = source_path + '/' + "matrix"
data_matrix_file_name = "%s/%s_%s.csv" %(data_matrix_files_name, "火车东站", "0116")

dataframe = pandas.read_csv(data_matrix_file_name,encoding='utf-8',header=0,index_col=0)

test_mae_list = {}
# 选择数据集
for i in range(2):
    #     选择滤波器
    for j in range(2):
        # 选择训练网络
        for k in range(2):
            dataset = dataframe.iloc[:, -2].values
            feature = dataframe.iloc[:, :-4].values
            # split into train and test
            train_size = int(len(dataset) * 0.8)
            sub_size = int(len(dataset) * 0.2)
            test_size = len(dataset) - train_size
            look_back = 24  # maybe is 24h

            model_name = ""
            kalman = ""
            feature_name = ""


            if i == 0:
                feature_name = "timeSeries"
            else:
                feature_name = "complex"



            if j == 1:
                kalman = "Kalman"
                dataset = Kalman(dataset)


            else:
                kalman = "NoKalman"

            # normalize

            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train = dataset[:train_size]
            test = dataset[train_size-look_back:len(dataset)]
            print(len(train))
            if i == 1:
                train_f, test_f = feature[:train_size], feature[train_size-look_back:len(dataset)]
                trainX, trainY = creat_dataset(train, look_back, train_f)
                testX, testY = creat_dataset(test, look_back, test_f)
            else:
                trainX,trainY = creat_dataset1(train,look_back)
                testX,testY = creat_dataset1(test,look_back)

            K.clear_session()
            if k==0:
                model_name = "BP"

                # reshape input to be [sample,time steps,features]
                trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1]))
                testX = numpy.reshape(testX,(testX.shape[0],testX.shape[1]))

                model = Sequential()
                model.add(Dense(int(trainX.shape[1]*0.75),input_dim=trainX.shape[1],activation='relu'))
                model.add(Dense(int(trainX.shape[1]*0.75*0.75),activation='relu'))
                model.add(Dense(int(trainX.shape[1]*0.75*0.75),activation='relu'))
                model.add(Dense(1,activation='sigmoid'))  # 输出层

                #     参数设置
                model.compile(loss='mae', optimizer='adam', metrics=['mae'])  # 编译模型必须的两个参数

                # 训练模型
                history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
                # 测试模型
                cost = model.evaluate(testX, testY, batch_size=1, verbose=2)


            else:
                model_name = "RNN"

                # reshape input to be [sample,time steps,features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

                # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(int(trainX.shape[2]*0.75), input_dim=trainX.shape[2], activation='tanh'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='mae', optimizer='adam', metrics=['mae'])
                history = model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2)
                cost = model.evaluate(testX, testY, batch_size=1, verbose=2)

            test_mae_list[model_name+'_'+kalman+'_'+feature_name]=cost[0]

#                 正式代码







print(test_mae_list)
output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'testoutput'))
# 保存history文件
json_file = open(output_path + '1'+'.json', 'w', encoding='utf-8')
json_file.write(json.dumps(test_mae_list))
json_file.close()