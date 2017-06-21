# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:merge_2.py

@time:2017/6/7 10:05

@desc:用于整合merge与FNNtest2
将结合 跑模型 和 整合文件

"""

import numpy
import os
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import json
from keras import backend as K


def creat_dataset(dataset, look_back, feature):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        f = feature[i + look_back]
        d = dataset[i:(i + look_back)]
        dataX.append(numpy.append(f, d))
        dataY.append(dataset[i + look_back])

    return numpy.array(dataX), numpy.array(dataY)


def creat_dataset1(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        dataX.append(dataset[i:(i + look_back)])
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


def Kalman(dataset):
    mean = numpy.mean(dataset)
    covariance = numpy.cov(dataset)
    kf = KalmanFilter(n_dim_obs=1, n_dim_state=1, initial_state_mean=mean, initial_state_covariance=covariance,
                      transition_matrices=[1], observation_matrices=[1], observation_covariance=covariance,
                      transition_covariance=numpy.eye(1), transition_offsets=None)
    filter = kf.filter(dataset)[0]
    return numpy.reshape(filter, (1, -1))[0]


# load dataset
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
data_matrix_files_name = source_path + '/' + "matrix"
data_matrix_file_name = "%s/%s_%s.csv" % (data_matrix_files_name, "凤起路", "0111")

dataframe = pandas.read_csv(data_matrix_file_name, encoding='utf-8', header=0, index_col=0)
dataset = dataframe.iloc[:, -2].values
XT = dataframe.iloc[:, :-4]

XT['实际人数'] = pandas.DataFrame(numpy.reshape(dataset, (-1, 1)), index=XT.index)
dataset = Kalman(dataset)
XT['滤波人数'] = pandas.DataFrame(numpy.reshape(dataset, (-1, 1)), index=XT.index)
train_size = int(len(dataset) * 0.8)#训练数据为前80%
test_size = len(dataset) - train_size#测试数据为后20%
tag = [0]*train_size
tag.extend([1]*test_size)
XT['测试集'] = pandas.DataFrame(numpy.reshape(tag, (-1, 1)), index=XT.index)

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
                feature_name = ""
            else:
                feature_name = "F"

            if j == 1:
                kalman = "YES"
                dataset = Kalman(dataset)


            else:
                kalman = "NO"

            # normalize

            scaler = MinMaxScaler(feature_range=(0, 1))
            dataset = scaler.fit_transform(dataset)
            train = dataset[:train_size]
            test = dataset[train_size - look_back:len(dataset)]
            print(len(train))
            if i == 1:
                train_f, test_f = feature[:train_size], feature[train_size - look_back:len(dataset)]
                trainX, trainY = creat_dataset(train, look_back, train_f)
                testX, testY = creat_dataset(test, look_back, test_f)
            else:
                trainX, trainY = creat_dataset1(train, look_back)
                testX, testY = creat_dataset1(test, look_back)

            K.clear_session()
            if k == 0:
                model_name = "BP"

                # reshape input to be [sample,time steps,features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1]))
                testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1]))

                model = Sequential()
                model.add(Dense(int(trainX.shape[1] * 0.75), input_dim=trainX.shape[1], activation='relu'))
                model.add(Dense(int(trainX.shape[1] * 0.75 * 0.75), activation='relu'))
                model.add(Dense(int(trainX.shape[1] * 0.75 * 0.75), activation='relu'))
                model.add(Dense(1, activation='sigmoid'))  # 输出层

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
                model.add(LSTM(int(trainX.shape[2] * 0.75), input_dim=trainX.shape[2], activation='tanh'))
                model.add(Dense(1, activation='sigmoid'))
                model.compile(loss='mae', optimizer='adam', metrics=['mae'])
                history = model.fit(trainX, trainY, nb_epoch=50, batch_size=1, verbose=2)
                cost = model.evaluate(testX, testY, batch_size=1, verbose=2)

            test_mae_list[feature_name + model_name + '_' + kalman] = {'history': history.history,
                                                                       'test': {'loss': cost[0],
                                                                                'mean_absolute_error': cost[1]}}

            trainPredict = model.predict(trainX)
            trainPredict = scaler.inverse_transform(trainPredict)
            testPredict = model.predict(testX)
            testPredict = scaler.inverse_transform(testPredict)
            print(testPredict)
            predict = numpy.row_stack((trainPredict, testPredict))
            predict = numpy.row_stack((numpy.zeros((look_back, 1)), predict))
            predict_list = predict.tolist()
            XT[feature_name + model_name + '_' + kalman] = pandas.DataFrame(numpy.reshape(predict_list, (-1, 1)), index=XT.index)




print(XT)
print(test_mae_list)
json_and_csv_output = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'app', 'static', 'output'))

# 保存history文件
json_file = open(json_and_csv_output + '/history_' + '0111' + '.json', 'w', encoding='utf-8')
json_file.write(json.dumps(test_mae_list))
json_file.close()



# 生成csv用于d3.js显示
# 保存矩阵 路径为 output/station_num.csv
XT.to_csv(path_or_buf=json_and_csv_output + '/' + '0111' + '.csv',
          index_label='datetime', encoding='utf-8')