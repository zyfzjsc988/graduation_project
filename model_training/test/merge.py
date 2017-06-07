# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file：merge.py

@time:2017/5/29 12:00

@desc:将8个模型的训练结果汇成两个文件：{csv,json}json保存训练过程信息，csv保存预测序列

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
    # smooth = kf.smooth(dataset)[0]
    # plt.plot(dataset,color='r')
    # plt.plot(filter,color='g')
    # plt.plot(smooth,color='b')
    # plt.show()
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
source_path = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'source'))
data_matrix_files_name = source_path + '/' + "matrix"
data_matrix_file_name = "%s/%s_%s.csv" %(data_matrix_files_name, "火车东站", "0116")

output_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'testoutput'))
json_and_csv_output = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,os.pardir,'app','static','output'))

BP_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"BP_NO"))
BP_YES =load_json_file("%s/history_0116_%s.json"%(output_path,"BP_YES"))
RNN_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"RNN_NO"))
RNN_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"RNN_YES"))
FBP_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"FBP_NO"))
FBP_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"FBP_YES"))
FRNN_NO = load_json_file("%s/history_0116_%s.json"%(output_path,"FRNN_NO"))
FRNN_YES = load_json_file("%s/history_0116_%s.json"%(output_path,"FRNN_YES"))


list=[]
list.append(BP_YES)
list.append(FBP_YES)
list.append(BP_NO)
list.append(FBP_NO)
list.append(RNN_YES)
list.append(FRNN_YES)
list.append(RNN_NO)
list.append(FRNN_NO)

# plt train
# plt.figure(1)#创建图表1
# plt.xlabel('epoch')
# plt.ylabel('mean_absolute_error')
# plt.title('Training')
# # shift train predictions for plotting
# style = ['^','s']
# color = ['g','b']
# line =['-','--']
# for i in range(2):
#     for j in range(2):
#         for d in range(2):
#             plt.plot(list[i*4+j*2+d]['history']['mean_absolute_error'],
#                      color=color[d],marker=style[i],linestyle=line[j],linewidth=2,
#                      label=str(i*4+j*2+d+1))
# plt.legend()
# plt.show()

# plt test
plt.figure(2)
test_mae = []
train_mae = []
index = np.arange(len(list))
barwidth = 0.2
opacity = 0.4
for i in range(len(list)):
    test_mae.append(list[i]['test']['mean_absolute_error'])
    train_mae.append(list[i]['history']['mean_absolute_error'][-1])



rect1 = plt.bar(index,test_mae,barwidth,alpha=opacity,color='b',label='Test Set')
rect2 = plt.bar(index+barwidth,train_mae,barwidth,alpha=opacity,color='r',label='Train Set')
print(train_mae)
plt.xlabel('model')
plt.ylabel('Test mean_absolute_error')
plt.title('Parameter comparing')

plt.xticks(index+barwidth,np.arange(1,len(list)+1))
plt.legend()
plt.show()







history_dict = {
    'BP_NO':{'history':BP_NO['history'],'test':BP_NO['test']},
    'BP_YES':{'history':BP_YES['history'],'test':BP_YES['test']},
    'RNN_NO':{'history':RNN_NO['history'],'test':RNN_NO['test']},
    'RNN_YES':{'history':RNN_YES['history'],'test':RNN_YES['test']},
    'FRNN_NO':{'history':FRNN_NO['history'],'test':FRNN_NO['test']},
    'FRNN_YES':{'history':FRNN_YES['history'],'test':FRNN_YES['test']},
    'FBP_NO':{'history':FBP_NO['history'],'test':FBP_NO['test']},
    'FBP_YES':{'history':FBP_YES['history'],'test':FBP_YES['test']},
}
# 保存history文件
json_file = open(json_and_csv_output + '/history_' + '0116' + '.json', 'w', encoding='utf-8')
json_file.write(json.dumps(history_dict))
json_file.close()





dataframe = pd.read_csv(data_matrix_file_name,encoding='utf-8',header=0,index_col=0)
dataset  = dataframe.iloc[:,-2].values
XT = dataframe.iloc[:, :-4]

XT['实际人数'] = pd.DataFrame(np.reshape(dataset, (-1, 1)), index=XT.index)
dataset = Kalman(dataset)
XT['滤波人数'] = pd.DataFrame(np.reshape(dataset, (-1, 1)), index=XT.index)
train_size = int(len(dataset) * 0.8)#训练数据为前80%
test_size = len(dataset) - train_size#测试数据为后20%
tag = [0]*train_size
tag.extend([1]*test_size)
XT['测试集'] = pd.DataFrame(np.reshape(tag, (-1, 1)), index=XT.index)

XT['BP_NO'] = pd.DataFrame(np.reshape(BP_NO['predict'], (-1, 1)), index=XT.index)
XT['BP_YES'] = pd.DataFrame(np.reshape(BP_YES['predict'], (-1, 1)), index=XT.index)
XT['RNN_NO'] = pd.DataFrame(np.reshape(RNN_NO['predict'], (-1, 1)), index=XT.index)
XT['RNN_YES'] = pd.DataFrame(np.reshape(RNN_YES['predict'], (-1, 1)), index=XT.index)
XT['FBP_NO'] = pd.DataFrame(np.reshape(FBP_NO['predict'], (-1, 1)), index=XT.index)
XT['FBP_YES'] = pd.DataFrame(np.reshape(FBP_YES['predict'], (-1, 1)), index=XT.index)
XT['FRNN_NO'] = pd.DataFrame(np.reshape(FRNN_NO['predict'], (-1, 1)), index=XT.index)
XT['FRNN_YES'] = pd.DataFrame(np.reshape(FRNN_YES['predict'], (-1, 1)), index=XT.index)



print(XT)

# 生成csv用于d3.js显示
# 保存矩阵 路径为 output/station_num.csv
XT.to_csv(path_or_buf=json_and_csv_output + '/' + '0116' + '.csv',
          index_label='datetime', encoding='utf-8')