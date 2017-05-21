# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:__init__.py.py

@time:2017/5/21 16:45

@desc:

"""

from forcast_web import Modelgrade
# 使用查询语句 filter 为过滤器
query = Modelgrade.query.filter_by(losstype = 'loss').all()
print(query)