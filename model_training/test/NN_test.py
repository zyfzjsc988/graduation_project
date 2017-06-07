# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:test——NN.py

@time:2017/5/27 9:51

@desc:形成 时序特征 训练结果 4个模型

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


def creat_dataset(dataset,look_back = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return numpy.array(dataX),numpy.array(dataY)

def Kalman(dataset):
    mean = numpy.mean(dataset)
    covariance = numpy.cov(dataset)
    kf = KalmanFilter(n_dim_obs=1,n_dim_state=1,initial_state_mean=mean,initial_state_covariance=covariance,
                      transition_matrices=[1],observation_matrices=[1],observation_covariance=covariance,
                      transition_covariance=numpy.eye(1),transition_offsets=None)
    filter = kf.filter(dataset)[0]
    # smooth = kf.smooth(dataset)[0]
    # plt.plot(dataset,color='r')
    # plt.plot(filter,color='g')
    # plt.plot(smooth,color='b')
    # plt.show()
    return numpy.reshape(filter,(1,-1))[0]

K.clear_session()
#load dataset
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
data_matrix_files_name = source_path + '/' + "matrix"
data_matrix_file_name = "%s/%s_%s.csv" %(data_matrix_files_name, "火车东站", "0116")


dataframe = pandas.read_csv(data_matrix_file_name,encoding='utf-8',header=0,index_col=0)
dataset  = dataframe.iloc[:,-2].values


# filter

# dataset = Kalman(dataset)
print(dataset)

# normalize

scaler = MinMaxScaler(feature_range=(0,1))
dataset = scaler.fit_transform(dataset)

# split into train and test
train_size = int(len(dataset)*0.8)
test_size = len(dataset)-train_size
look_back = 24 # maybe is 24h
train ,test= dataset[0:train_size] , dataset[train_size-look_back:len(dataset)]

# reshape
trainX,trainY = creat_dataset(train,look_back)
testX,testY = creat_dataset(test,look_back)


# # reshape input to be [sample,time steps,features]
trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1],1))
testX = numpy.reshape(testX,(testX.shape[0],testX.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(18,input_dim= 1,activation='tanh'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='mae',optimizer='adam',metrics=['mae'])
history = model.fit(trainX,trainY,nb_epoch=30,batch_size=1,verbose=2)
cost = model.evaluate(testX,testY,batch_size=1,verbose=2)

#
#
# # reshape input to be [sample,time steps,features]
# trainX = numpy.reshape(trainX,(trainX.shape[0],trainX.shape[1]))
# testX = numpy.reshape(testX,(testX.shape[0],testX.shape[1]))
#
# model = Sequential()
# model.add(Dense(18,input_dim=24,activation='relu'))
# model.add(Dense(14,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))  # 输出层
#
# #     参数设置
# model.compile(loss='mae', optimizer='adam', metrics=['mae'])  # 编译模型必须的两个参数
#
# # 训练模型
# history = model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
# # 测试模型
# cost = model.evaluate(testX, testY, batch_size=1, verbose=2)







# inverse
trainPredict = model.predict(trainX)
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = model.predict(testX)
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# # calculate root mse
# trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
# print('Train Score:%.2f RMSE'%(trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
# print('Test Score:%.2f RMSE'%(testScore))
print(history.history)
print(cost)

input('input:')
history_dict = {'history':history.history}
history_dict['test'] = {}
for i in range(len(model.metrics_names)):
     history_dict['test'][model.metrics_names[i]] = cost[i]
predict = numpy.row_stack((trainPredict,testPredict))
predict = numpy.row_stack((numpy.zeros((look_back,1)),predict))
print(len(predict))
history_dict['predict'] = predict.tolist()
#
# output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'testoutput'))
# # 保存history文件
# json_file = open(output_path + '/history_' + '0116_BP_NO' + '.json', 'w', encoding='utf-8')
# json_file.write(json.dumps(history_dict))
# json_file.close()



plt.plot(history.history['loss'])
plt.show()




# plot
# shift train predictions for plotting
# trainPredictPlot = numpy.empty_like(dataset)
# trainPredictPlot[:] = numpy.nan
# for i in range(len(trainPredict)):
#     trainPredictPlot[i+look_back] = trainPredict[i,0]
# # shift test predictions for plotting
# testPredictPlot = numpy.empty_like(dataset)
# testPredictPlot[:] = numpy.nan
# for i in range(len(testPredict)):
#     testPredictPlot[i+len(trainPredict)+2*look_back] = testPredict[i,0]
# plt.scatter(range(len(dataset)),scaler.inverse_transform(dataset))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()


