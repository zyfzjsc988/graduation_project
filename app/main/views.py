# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:views.py

@time:2017/5/22 15:47

@desc:
蓝本中定义的程序业务逻辑路由
"""
from flask import render_template, session, redirect, url_for, current_app,g
from ..models import Modelgrade
from . import main
from .forms import SelectForm
from ..train_result import Predict
matrix_file_path = "matrix"
history_file_path = "output/history"
model_file_path = "output/model"
weights_file_path = "output/weights"


@main.before_app_first_request
def bf_app_request():
    session['log'] = []


# 主页，路由装饰器由蓝本提供所以 从main出发
@main.route('/', methods=['GET', 'POST'])
def index():
    form = SelectForm()
    form.place.choices = []
    query_result = Modelgrade.query
    for item in query_result.all():
        if (item.placeno,item.placename) not in form.place.choices:
            form.place.choices.append((item.placeno,item.placename))

    if form.validate_on_submit():
        session['log'] = []
        # 提交表单后重定向到index函数
        session['place'] = form.place.data
        session['type'] = form.type.data
        # show data
        query_by_place = query_result.filter(Modelgrade.placeno == form.place.data).order_by(form.type.data)
        for item in query_by_place.all():
            matrix_url = url_for('static',
                                 filename="%s/%s_%s.csv"%(matrix_file_path,item.placename,item.placeno),
                                 _external=True)
            history_url = url_for('static',
                                  filename="%s_%s_%s_%s_%s_%s_%d.json"%(history_file_path,
                                                                        item.placeno,
                                                                        item.nnname,
                                                                        item.losstype,
                                                                        item.opttype,
                                                                        item.activation,
                                                                        item.dropout),
                                  _external=True)
            model_url = url_for('static',
                                  filename="%s_%s_%s_%s_%s_%s_%d.json" % (model_file_path,
                                                                          item.placeno,
                                                                          item.nnname,
                                                                          item.losstype,
                                                                          item.opttype,
                                                                          item.activation,
                                                                          item.dropout),
                                  _external=True)
            weights_url = url_for('static',
                                  filename="%s_%s_%s_%s_%s_%s_%d.h5" % (weights_file_path,
                                                                          item.placeno,
                                                                          item.nnname,
                                                                          item.losstype,
                                                                          item.opttype,
                                                                          item.activation,
                                                                          item.dropout),
                                  _external=True)

            predict = Predict(matrix_url,history_url,model_url,weights_url)
            model, X, predicted, Y =predict.train_and_predict()
            session['log'].append(len(X))
            session['log'].append(len(predicted))
            session['log'].append(len(Y))




        # 在蓝本中需要url_for用（蓝本名.函数名）所以main.index 简写为.index
        return redirect(url_for('.index'))

    return render_template('show_all_flask.html', log=session.get('log'),
                           form=form, place=session.get('place'),
                           type=session.get('type'))

