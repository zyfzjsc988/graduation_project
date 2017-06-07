# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:create_csv.py

@time:2017/5/17 21:58

@desc:
input : files(hz**.dat/weather_HZ.json/config.json)

output: files(station_name_NO.csv)
        csv   [time,weather[],isholiday,isweekends,
                isdayhightime,isnighthightime,last_day_people_count,
                last_7day_people_count,people_count]

time cell (an hour)

"""

import json
import os
import time
import datetime
import pandas as pd
import numpy as np
import random

def load_json_file(filename):
    """
    decoding json文件
    :param filename: file path
    :return: dict或list
    """
    json_file = open(filename,'r',encoding='utf-8')
    return json.load(json_file)

def load_dat_file(filename):
    """
    打开dat文件，将其转换成dict
    :param filename: dat文件Path
    :return: dict
    """

    def date_to_time_y2s(str):
        # 将str转换成datetime
        t = time.strptime(str, "%Y-%m-%dT%H:%M:%S.000Z")
        y, m, d, hh, mm, ss = t[0:6]
        return datetime.datetime(y, m, d, hh, mm, ss)

    file = open(filename, 'r', encoding='utf-8')
    ticket_dict = {}
    for line in file:
        js = json.loads(line)
        if 'voucherTime' in js:
            # 获得可能需要用到的数据
            ID = js['_id']['$numberLong']  # db_id
            userId = js['userId']  # userId 用户id
            sNum = js['startNum']  # startNum 起点站id
            sName = js['startName']  # startName 起点站名
            eNum = js['endNum']  # endNum 终点站id
            eName = js['endName']  # endName 终点站名
            cTime = js['voucherTime']['$date']  # voucherTime 核销时间
            ctDate = date_to_time_y2s(cTime)  # datetime:y,m,d,hh,mm,ss 将核销时间转换成date和time
            counts = js['counts']  # ticket_count 购票总数

            # 存储到dict中
            ticket_dict[ID] = {'userId': userId, 'startNum': sNum, 'startName': sName,
                               'endNum': eNum, 'endName': eName, 'voucherTime': ctDate.strftime('%Y-%m-%d %X'),
                               'counts': counts}
    return ticket_dict

def find_all_station(json_file):
    """
    构建所有起点站的站名list并返回
    :param json_flie: 
    :return: dict
    """
    station_list = {}
    for id in json_file.keys():
        start_num = json_file[id]['startNum']
        if start_num not in station_list:
            station_list[start_num] = json_file[id]['startName']
    return station_list

def find_all_weather(json_file):
    """
    构建weather list并返回
    :param json_flie: dict类型
    :return: list
    """
    weather_list = []
    for item in json_file.keys():
        if json_file[item] not in weather_list:
            weather_list.append(json_file[item])

    return weather_list

def save_json(filename,json_arr):
    """
    存储json到文件
    :param filename: 
    :param json_arr: 
    :return: 
    """
    json_str = json.dumps(json_arr, ensure_ascii=False)
    jsonfile = open(filename, 'w', encoding='utf-8')
    jsonfile.write(json_str)
    jsonfile.close()

def change_list_to_dict(listname):
    weatherdict = {}
    i = 0
    while i < len(listname):
        weatherdict[listname[i]['date']]=listname[i]['weather']
        i+=1
    return weatherdict


def count_people(date,hour,data):
    count = 0
    for k in data.keys():
        time_str = data[k]['voucherTime']
        t = time.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        if t[0] == date.year and t[1] == date.month and t[2] == date.day and t[3] == hour:
            count += data[k]['counts']
    return count


if __name__=="__main__":

    # 数据源路径，使用本地访问需要使用绝对地址才能够访问
    source_path = os.path.abspath(os.path.join(os.path.dirname(__file__),os.pardir,'source'))
    # flow_file_name = "hzmetro0402.dat"
    # weather_file_name = "weather_HZ.json"
    config_file_name = "config.json"


    #加载json和dat文件，将其结构化为dict或list
    config = load_json_file(source_path+'/'+config_file_name)#dict
    # 从config中读取路径
    flow_file_name = config['flow_file_name']
    flow_dict = load_dat_file(source_path+'/'+flow_file_name)#dict
    weather_file_name = config['weather_file_name']
    weather_dict = change_list_to_dict(load_json_file(source_path+'/'+weather_file_name))#dict

    """
    # 创建起点站dict并且搜索得到所有起点站
    start_station_dict = find_all_station(flow_dict)

    # 创建weather list并且搜索得到所有weather
    weather_list = find_all_weather(weather_dict)#list
    # 存储入config中
    config['station'] = start_station_dict
    config['weather'] = weather_list
    save_json(source_path+'/'+config_file_name,config)
    """
    # 数据起止时间
    start_time = datetime.datetime.strptime(config['start_time'], '%Y-%m-%d')
    end_time = datetime.datetime.strptime(config['end_time'], '%Y-%m-%d')
    # 构建矩阵numpy（以一小时为时间粒度）从start time 到 end time
    # 矩阵的标头list ：[time,weather[],isholiday,isweekends,
    #             isdayhightime,isnighthightime,last_day_people_count,
    #             last_7day_people_count,people_count]
    title = ["假日", "周末", "早高峰", "晚高峰"]
    title.extend(config['weather'])
    title.extend(["b7", "b1", "人数", "测试集"])

    # 对于每个站点构建该站点的矩阵并保存
    # for 每一个站点名，选出该起点站，添加到该站点中的dict中，再对该dict进行操作
    for startNum in config['station']:
        stationName = config['station'][startNum]
        each_station_flow_dict = {}
        for id in flow_dict.keys():
            if flow_dict[id]['startNum'] == startNum:
                each_station_flow_dict[id] = flow_dict[id]
        # # 对订单信息先进行根据核销时间的排序
        # sort_list = sorted(each_station_flow_dict.items(), key=lambda each_station_flow_dict: each_station_flow_dict[1]['voucherTime'])
        # 新的用于统计存储某站点的人流数的dict，主要用于构建矩阵时的人流量查询{date:time:count}
        count_time_dict = {}
        matrixs_ndarray = []#矩阵
        date_list = [] #矩阵的第一列 日期
        begin = start_time
        while begin <= end_time:
            #    每天 天气情况 假日 周末
            begin_str = begin.strftime('%Y-%m-%d')
            weather = weather_dict[begin_str]
            holiday = begin_str in config['holiday']
            weekends = (begin.weekday() == 5 or begin.weekday() == 6) and begin_str not in config['not_weekends']

            #           每小时 早晚高峰限行时间 前7天同时刻人数 当前时刻人数
            count_dict = {}
            rand = int(random.random() * 24)
            # 遍历一天24小时
            for h in range(24):
                # 统计人流
                people_counting = count_people(begin,h,each_station_flow_dict)
                count_dict[h] = people_counting
                count_time_dict[begin_str] = count_dict
                # 将超过起始日期的七天后的人流数 构建并添加到矩阵中
                if begin - datetime.timedelta(days=7) >=start_time:
                    date_and_time = begin.replace(hour=h)
                    date_list.append(date_and_time)#添加行标签
                    # 一小时一行数据
                    all_data_list = []
                    # 是否是假日
                    if holiday:
                        all_data_list.append(1)
                    else:
                        all_data_list.append(0)
                    # 是否是周末
                    if weekends:
                        all_data_list.append(1)
                    else:
                        all_data_list.append(0)
                    # 是否早高峰
                    if h in config['m_high_time']:
                        all_data_list.append(1)
                    else:
                        all_data_list.append(0)
                    # 是否晚高峰
                    if h in config['n_high_time']:
                        all_data_list.append(1)
                    else:
                        all_data_list.append(0)

                    for item in config['weather']:
                        if weather == item :
                            all_data_list.append(1)
                        else:
                            all_data_list.append(0)
                    b7_count = count_time_dict[(begin - datetime.timedelta(days=7)).strftime('%Y-%m-%d')][h]
                    all_data_list.append(b7_count)
                    b1_count = count_time_dict[(begin - datetime.timedelta(days=1)).strftime('%Y-%m-%d')][h]
                    all_data_list.append(b1_count)
                    all_data_list.append(people_counting)
                    if rand == h:
                        all_data_list.append(1)
                    else:
                        all_data_list.append(0)
                    matrixs_ndarray.append(all_data_list)#将每行结果添加到矩阵中

            begin += datetime.timedelta(days=1)
        # 使用pandas format   index行标签 columus列标签
        df = pd.DataFrame(np.array(matrixs_ndarray), index=date_list, columns=title)
        # 保存矩阵 路径为 source/matrix/station_name_NO.csv
        df.to_csv(path_or_buf=source_path+'/'+"matrix"+'/'+stationName+'_'+startNum+'.csv',index_label='datetime',encoding='utf-8')
