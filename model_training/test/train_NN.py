# -*- coding: utf-8 -*-

"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:train_NN.py

@time:2017/5/18 19:58

@desc:
最初版本，十分没有灵活性
train neural network
NN = BP / elman
损失函数 loss = [mse,mae,mape,msle,binary_crossentropy,categorical_crossentropy] 
学习算法、优化器 optimizer = [SGD,RMSprop,Adagrad,Adadelta,Adam,Adamax]
评估指标 MSE/MAE

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import json
from keras.models import Sequential,model_from_json
from keras.layers import Dense,Dropout,LSTM
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine,Column,String,Integer,Float,ForeignKey,func
from sqlalchemy.orm import sessionmaker,relationship
from sqlalchemy.ext.declarative import declarative_base
from pykalman import KalmanFilter
from keras import backend as K


# 对象的基类
Base = declarative_base()


class Modelinfo(Base):
    __tablename__='modelinfo'
#     表结构

    id = Column(Integer, primary_key=True)
    modelname = Column(String(256),
                       ForeignKey('modelhistory.modelname',ondelete='CASCADE', onupdate='CASCADE'))
    trainloss = Column(Float)
    trainaccuracy = Column(Float)
    testloss = Column(Float)
    testaccuracy = Column(Float)
#
# class Modelhistory(Base):
#     __tablename__ = 'modelhistory'
#     #     表结构
#
#     id = Column(Integer, primary_key=True)
#     modelname = Column(String(256))
#     epoch = Column(Integer)
#     loss = Column(Float)
#     accuracy = Column(Float)
#
#     #   一对多关系
#     models = relationship('Modelinfo')


mysql_engine = create_engine("mysql://root:password@localhost:3306/flownn")
DBsession = sessionmaker(bind=mysql_engine)
session = DBsession()

def add_model_to_db(model_name,history):

    id = session.query(func.max(Modelinfo.id)).one()[0]
    if id :
        if id == 0:
            id = 1
        else:
            id+=1
    else:
        id = 1

    train_loss = history['loss'][-1]
    train_acc = history['acc'][-1]
    test_loss = history['test']['loss']
    test_acc = history['test']['acc']
    new_model = Modelinfo(id=id, modelname=model_name,
                              trainloss=train_loss, trainaccuracy=train_acc,
                              testloss=test_loss, testaccuracy=test_acc)

    session.add(new_model)
    session.commit()

# def add_history_to_db(model_name,history):
#     epoch = len(history['loss'])
#
#     id = session.query(func.max(Modelhistory.id)).one()[0]
#     if id :
#         if id == 0:
#             id = 1
#         else:
#             id += 1
#     else:
#         id = 1
#
#     session.execute(
#         Modelhistory.__table__.insert(),
#         [{'id':id+x,'modelname':model_name,'epoch':x+1,'loss':history['loss'][x],'accuracy':history['acc'][x]} for x in range(0,epoch)]
#     )
#     session.commit()
def save_json(filename,json_arr):
    """
    存储json到文件
    :param filename: 
    :param json_arr: 
    :return: 
    """
    if type(json_arr) == str:
        json_str = json_arr
    else:
        json_str = json.dumps(json_arr, ensure_ascii=False)
    jsonfile = open(filename, 'w', encoding='utf-8')
    jsonfile.write(json_str)
    jsonfile.close()

def data_normalizing(np):
    """
    将数据进行区间缩放，缩放到[0,1]的范围中
    :param np: 原矩阵
    :return: 缩放后的矩阵 区间大小，区间最小值
    """
    min_max_scaler = MinMaxScaler(feature_range=(0,1))
    fix = min_max_scaler.fit_transform(np)
    scaler = MinMaxScaler(feature_range=(0,1))
    y_fix = scaler.fit_transform(np[:,-1])
    return fix,scaler

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



def create_BP(loss_name,opt_name,act_name,X_train, Y_train, X_test, Y_test,epoch):
    """
    BP神经网络结构
    :param loss_name: 
    :param opt_name: 
    :return: 
    """

    mse = 100
    for k in range(2):
        model = Sequential()
        model.add(Dense(6, input_dim=24, activation=act_name))
        model.add(Dense(6, activation=act_name))
        model.add(Dense(6, activation=act_name))
        model.add(Dense(1))  # 输出层

        #     参数设置
        model.compile(loss=loss_name, optimizer=opt_name, metrics=['mae'])  # 编译模型必须的两个参数

        # 训练模型
        history = model.fit(X_train, Y_train, epochs=epoch, batch_size=1, verbose=2)
        # 测试模型
        cost = model.evaluate(X_test, Y_test, batch_size=1, verbose=2)

        # 保存
        hist = history.history
        hist['test'] = {}
        for i in range(len(model.metrics_names)):
            hist['test'][model.metrics_names[i]] = cost[i]
        if mse > hist['mean_absolute_error'][-1]:
            mse = hist['mean_absolute_error'][-1]
            new_model = model
            new_hist = hist
        if k < 9:
            K.clear_session()

    return new_hist,new_model


def create_ELMAN(loss_name,opt_name,act_name,X_train,Y_train,X_test,Y_test,epoch):
    mse = 100
    for k in range(2):
        # create and fit the LSTM network
        model = Sequential()
        model.add(LSTM(6,input_dim = 1))
        model.add(Dense(1,activation=act_name))
        model.compile(loss=loss_name,optimizer=opt_name,metrics=['mae'])

        history = model.fit(X_train,Y_train,nb_epoch=epoch,batch_size=1,verbose=2)
        cost = model.evaluate(X_test,Y_test,batch_size=1,verbose=2)

        # 保存
        hist = history.history
        hist['test'] = {}
        for i in range(len(model.metrics_names)):
            hist['test'][model.metrics_names[i]] = cost[i]

        print()
        if mse > hist['mean_absolute_error'][-1]:
            print(mse)
            mse = hist['mean_absolute_error'][-1]
            new_model = model
            new_hist = hist
        K.clear_session()
    return new_hist, new_model



def create_FBP(loss_name,opt_name,act_name,X_train, Y_train, X_test, Y_test,epoch):
    """
    BP神经网络结构
    :param loss_name: 
    :param opt_name: 
    :return: 
    """
    mse = 100
    for k in range(2):
        dim = len(X_train[0])
        model = Sequential()
        model.add(Dense(40,input_dim=dim,activation=act_name))#Dense全连接层，input有18维
        model.add(Dense(40,activation=act_name))
        model.add(Dense(40,activation=act_name))
        model.add(Dense(1))#输出层

        #     参数设置
        model.compile(loss=loss_name,optimizer=opt_name,metrics=['mae'])#编译模型必须的两个参数
        # 训练模型
        history = model.fit(X_train, Y_train, epochs=epoch, batch_size=1,verbose=2)
        # 测试模型
        cost = model.evaluate(X_test, Y_test, batch_size=1,verbose=2)
        # 保存
        hist = history.history
        hist['test'] = {}
        for i in range(len(model.metrics_names)):
            hist['test'][model.metrics_names[i]] = cost[i]

        print()
        if mse > hist['mean_absolute_error'][-1]:
            mse = hist['mean_absolute_error'][-1]
            new_model = model
            new_hist = hist
        K.clear_session()
    return new_hist,new_model


def predict_point_by_point(model,data,scaler):
    #直接预测
    predicted  = model.predict(data)
    predicted = scaler.inverse_transform(predicted)

    return predicted

def plot_results(predicted_data,true_data):
    plt.plot(true_data,label= 'P True DATA')
    plt.plot(predicted_data,label = 'Prediction')
    plt.legend()
    plt.show()

#
# def load_and_display(station_num,NN_name,loss_name,opt_name,dropout,act_name):
#
#     output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'output'))
#     if dropout:
#         isdropout = 1
#     else:
#         isdropout = 0
#     history_file_name = "%s/history_%s_%s_%s_%s_%s_%d.json" % (
#         output_path, station_num, NN_name, loss_name, opt_name, act_name, isdropout)
#     model_json_file = "%s/model_%s_%s_%s_%s_%s_%d.json" % (
#         output_path, station_num, NN_name, loss_name, opt_name, act_name, isdropout)
#     model_weights_file = "%s/weights_%s_%s_%s_%s_%s_%d.h5" % (
#         output_path, station_num, NN_name, loss_name, opt_name, act_name, isdropout)
#
#     model = model_from_json(open(model_json_file,'r',encoding='utf-8').read())
#     model.load_weights(model_weights_file)
#     history = json.load(open(history_file_name,'r',encoding='utf-8'))
#     print(model.to_json())
#     print(history)
#     return model


def creat_dataset(dataset,look_back = 1):
    dataX,dataY = [],[]
    for i in range(len(dataset)-look_back):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])
    return np.array(dataX),np.array(dataY)


def separate(dataset):
    dataX, dataY = [], []

    for i in range(len(dataset)):
        dataX.append(dataset[i,:-1])
        dataY.append(dataset[i,-1])
    return np.array(dataX),np.array(dataY)



def Kalman(dataset):
    mean = np.mean(dataset)
    covariance = np.cov(dataset)
    kf = KalmanFilter(n_dim_obs=1,n_dim_state=1,initial_state_mean=mean,initial_state_covariance=covariance,
                      transition_matrices=[1],observation_matrices=[1],observation_covariance=covariance,
                      transition_covariance=np.eye(1),transition_offsets=None)
    filter = kf.filter(dataset)[0]
    # smooth = kf.smooth(dataset)[0]
    # plt.plot(dataset,color='r')
    # plt.plot(filter,color='g')
    # plt.plot(smooth,color='b')
    # plt.show()
    return np.reshape(filter,(1,-1))

if __name__ == '__main__':

    # session.query(Modelinfo).delete()
    # session.commit()


    # 数据源地址
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
    station_dict = json.load(open(source_path+'/config.json','r',encoding='utf-8'))['station']
    data_matrix_files_name = source_path + '/' + "matrix"

    # NN_name: BP / ELMAN
    NN_name = "BP"
    # loss_name: MES / MAE / MAPE / MSLE / BC / logcosh
    # loss_name_list = ["mse", "mae", "mape", "msle", "binary_crossentropy", "logcosh"]
    loss_name = "msle"
    # opt_name: SGD / RMS / AG / AD / AM / AMAX
    # opt_name_list = ["sgd", "rmsprop", "adagrad", "adadelta", "adam", "adamax"]
    opt_name = "adam"
    # act_name: elu,relu,tanh,sigmoid,linear
    # act_name_list = ["elu", "relu", "tanh", "sigmoid", "linear"]
    act_name = "relu"
    epoch = 10


    for filename in station_dict:
        # 第一个循环，对每个矩阵文件
        history_dict = {}
        # source name 和 station num
        station_num = filename
        station_name = station_dict[filename]
        if station_name != '火车东站':
            continue
        # 提取矩阵
        data_matrix_file_name = "%s/%s_%s.csv" %(data_matrix_files_name, station_name, filename)
        data_matrix = pd.read_csv(data_matrix_file_name,encoding='utf-8',header=0,index_col=0)
        XT = data_matrix.iloc[:, :-4]
        target = data_matrix.values[:, :-1]  # 基于特征的BP神经网络，有效矩阵信息，剔除最后一列测试集标签，列信息【特征，真实人数】
        Y = target[:, -1]  # 实际人数
        dataset = data_matrix.iloc[:, -2].values #基于时间序列的BP/ELMAN神经网络的训练数据【真实人数】




        #数据预处理

        timeseries_scaler = MinMaxScaler(feature_range=(0,1))
        dataset = timeseries_scaler.fit_transform(dataset)

        train_size = int(len(dataset) * 0.8)#训练数据为前80%
        test_size = len(dataset) - train_size#测试数据为后20%

        """
        先进行无滤波的实验
        
        """

        # 实验一：普通BP
        look_back = 24  # maybe is 24h
        train, test = dataset[0:train_size], dataset[train_size-look_back:len(dataset)]

        # reshape
        look_back = 24  # maybe is 24h
        X_train, Y_train = creat_dataset(train, look_back)
        X_test, Y_test = creat_dataset(test, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        BP_history, BP_model = create_BP(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                   X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch = epoch)

        history_dict['BP-NO'] = BP_history
        BPtrain_predict = predict_point_by_point(BP_model, X_train,timeseries_scaler)
        BPtest_predict = predict_point_by_point(BP_model, X_test,timeseries_scaler)
        BP_predict = np.row_stack((BPtrain_predict,BPtest_predict))
        BP_predict = np.row_stack((np.zeros((look_back,1)),BP_predict))
        XT['BP-NO'] = pd.DataFrame(np.reshape(BP_predict, (-1, 1)), index=XT.index)

        # 实验二：普通ELMAN

        # reshape input to be [sample,time steps,features]
        X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))
        #
        # # # reshape input to be [sample,time steps,features]
        # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        ELMAN_history, ELMAN_model = create_ELMAN(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                         X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch=epoch)
        history_dict['ELMAN-NO'] = ELMAN_history
        Etrain_predict = predict_point_by_point(ELMAN_model, X_train, timeseries_scaler)
        Etest_predict = predict_point_by_point(ELMAN_model, X_test, timeseries_scaler)
        E_predict = np.row_stack((Etrain_predict,Etest_predict))
        E_predict = np.row_stack((np.zeros((look_back,1)),E_predict))
        XT['ELMAN-NO'] = pd.DataFrame(np.reshape(E_predict, (-1, 1)), index=XT.index)

        # 实验三：基于特征的BP神经网络

        # 数据预处理
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target = feature_scaler.fit_transform(target)  # 对数据进行放缩

        train, test = target[0:train_size], target[train_size:len(dataset)]
        X_train, Y_train = separate(train)
        X_test, Y_test = separate(test)
        FBP_history, FBP_model = create_FBP(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                         X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch=epoch)
        history_dict['FBP-NO'] = FBP_history
        FBPtrain_predict = predict_point_by_point(FBP_model, X_train, timeseries_scaler)
        FBPtest_predict = predict_point_by_point(FBP_model, X_test, timeseries_scaler)
        FBP_predict = np.row_stack((FBPtrain_predict,FBPtest_predict))
        XT['FBP-NO'] = pd.DataFrame(np.reshape(FBP_predict, (-1, 1)), index=XT.index)




        """
        预处理 kalman滤波
        """
        target = data_matrix.values[:, :-2]  # 基于特征的BP神经网络，有效矩阵信息，剔除最后一列测试集标签，列信息【特征，真实人数】
        dataset = data_matrix.iloc[:, -2].values  # 基于时间序列的BP/ELMAN神经网络的训练数据【真实人数】
        dataset = Kalman(dataset) #使用卡尔曼滤波处理数据
        XT['滤波人数'] = pd.DataFrame(np.reshape(dataset, (-1, 1)), index=XT.index)
        target = np.column_stack(((target,dataset.reshape(-1,1))))


        # 实验四：普通BP+KALMAN滤波

        #数据预处理

        dataset = dataset[0]
        timeseries_scaler = MinMaxScaler(feature_range=(0,1))
        dataset = timeseries_scaler.fit_transform(dataset)

        train, test = dataset[0:train_size], dataset[train_size-look_back:len(dataset)]

        # reshape
        X_train, Y_train = creat_dataset(train, look_back)
        X_test, Y_test = creat_dataset(test, look_back)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1]))
        BP_history, BP_model = create_BP(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                   X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch = epoch)
        history_dict['BP-YES'] = BP_history
        BPtrain_predict = predict_point_by_point(BP_model, X_train,timeseries_scaler)
        BPtest_predict = predict_point_by_point(BP_model, X_test,timeseries_scaler)
        BP_predict = np.row_stack((BPtrain_predict,BPtest_predict))
        BP_predict = np.row_stack((np.zeros((look_back,1)),BP_predict))
        XT['BP-YES'] = pd.DataFrame(np.reshape(BP_predict, (-1, 1)), index=XT.index)

        # 实验五：普通ELMAN+kalman

        # reshape input to be [sample,time steps,features]
        X_train = np.reshape(X_train, (X_train.shape[0], look_back, 1))
        X_test = np.reshape(X_test, (X_test.shape[0], look_back, 1))
        #
        # # # reshape input to be [sample,time steps,features]
        # X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        # X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        ELMAN_history, ELMAN_model = create_ELMAN(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                         X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch=epoch)
        history_dict['ELMAN-YES'] = ELMAN_history
        Etrain_predict = predict_point_by_point(ELMAN_model, X_train, timeseries_scaler)
        Etest_predict = predict_point_by_point(ELMAN_model, X_test, timeseries_scaler)
        E_predict = np.row_stack((Etrain_predict,Etest_predict))
        E_predict = np.row_stack((np.zeros((look_back,1)),E_predict))
        XT['ELMAN-YES'] = pd.DataFrame(np.reshape(E_predict, (-1, 1)), index=XT.index)

        # 实验六：基于特征的BP神经网络+kalman

        # 数据预处理
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target = feature_scaler.fit_transform(target)  # 对数据进行放缩

        train, test = target[0:train_size], target[train_size:len(dataset)]
        X_train, Y_train = separate(train)
        X_test, Y_test = separate(test)
        FBP_history, FBP_model = create_FBP(loss_name=loss_name, opt_name=opt_name, act_name=act_name,
                                         X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test,epoch=epoch)
        history_dict['FBP-YES'] = FBP_history
        FBPtrain_predict = predict_point_by_point(FBP_model, X_train, timeseries_scaler)
        FBPtest_predict = predict_point_by_point(FBP_model, X_test, timeseries_scaler)
        FBP_predict = np.row_stack((FBPtrain_predict,FBPtest_predict))
        print(len(FBP_predict))
        XT['FBP-YES'] = pd.DataFrame(np.reshape(FBP_predict, (-1, 1)), index=XT.index)
















        # # 用于BP特征的测试集标签
        # tag = data_matrix.values[:, -1]  # 训练集测试集标签




        XT['实际人数'] = pd.DataFrame(np.reshape(Y, (-1, 1)), index=XT.index)
        tag = [0]*train_size
        tag.extend([1]*test_size)
        XT['测试集'] = pd.DataFrame(np.reshape(tag, (-1, 1)), index=XT.index)





        # 生成csv用于d3.js显示
        # 保存矩阵 路径为 output/station_num.csv
        output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'output'))
        XT.to_csv(path_or_buf= output_path + '/' + station_num + '.csv',
                  index_label='time', encoding='utf-8')
        # 保存history文件
        json_file = open(output_path+'/history_'+station_num+'.json','w',encoding='utf-8')
        json_file.write(json.dumps(history_dict))
        json_file.close()


    session.close()