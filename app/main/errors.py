# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:errors.py

@time:2017/5/22 15:44

@desc:
错误处理路由
"""
from flask import render_template
from . import main

# 注册错误处理路由 app_errorhandler表示全局错误处理程序
@main.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@main.app_errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
