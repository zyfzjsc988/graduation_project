# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:views.py

@time:2017/5/22 15:47

@desc:
蓝本中定义的程序业务逻辑路由
"""
from flask import render_template, session, redirect, url_for, current_app
from .. import db
from ..models import Modelgrade
from . import main
from .forms import SelectForm

# 主页，路由装饰器由蓝本提供所以 从main出发
@main.route('/', methods=['GET', 'POST'])
def index():
    form = SelectForm()
    query_result = Modelgrade.query.all()
    log = ""
    for item in query_result:
        form.place.choices.append((item.placename, item.placename))
        form.type.choices.append((item.losstype, item.losstype))

    if form.validate_on_submit():
        # 提交表单后重定向到index函数
        session['place'] = form.place.data
        session['type'] = form.type.data
        # 在蓝本中需要url_for用（蓝本名.函数名）所以main.index 简写为.index
        return redirect(url_for('.index'))

    return render_template('show_all_flask.html', log=log,
                           form=form, place=session.get('place'),
                           type=session.get('type'))

