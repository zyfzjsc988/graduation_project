# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:__init__.py.py

@time:2017/5/21 16:47

@desc:创建蓝本，用于注册路由和error

"""
from flask import Blueprint
# 实例化创建蓝本 ：蓝本所在模块和名字
main = Blueprint('main', __name__)

# 在末尾导入依赖是为了避免循环导入依赖包，因为views和errors是依赖于main的
from . import views, errors