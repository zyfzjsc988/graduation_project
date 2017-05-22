# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:__init__.py

@time:2017/5/22 15:33

@desc:使用工厂模式创建app实例

"""
from flask import Flask
from flask_bootstrap import Bootstrap
from flask_sqlalchemy import SQLAlchemy
from config import config

bootstrap = Bootstrap()
db = SQLAlchemy()#单例模式

# 工厂函数
def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])#根据不同环境名，创造不同环境
    config[config_name].init_app(app)

    bootstrap.init_app(app)
    db.init_app(app)

    # 附加路由和自定义的错误页面
    # 使用蓝本定义路由，在蓝本中定义的路由处于休眠状态，注册到程序上后才可用
    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    print("success")
    return app

