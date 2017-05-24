# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:views.py

@time:2017/5/22 15:47

@desc:
蓝本中定义的程序业务逻辑路由
"""
from flask import render_template, session, redirect, url_for, current_app,g,make_response
from ..models import Modelhistory,Modelinfo,Place,db
from . import main
from .forms import SelectForm
import os
history_file_path = "output/history"

#
# @main.route('/<filepath>')
@main.route('/<path>')
def today(path):
    base_dir = os.path.dirname(__file__)
    resp = make_response(open(os.path.join(base_dir, path)).read())
    resp.headers["Content-type"]="application/json;charset=UTF-8"
    return resp

@main.before_app_first_request
def bf_app_request():
    session['log'] = []

# 主页，路由装饰器由蓝本提供所以 从main出发
@main.route('/', methods=['GET', 'POST'])
def index():

    form = SelectForm()
    form.place.choices = []
    for item in Place.query.all():
        form.place.choices.append((item.id,item.placename))


    if form.validate_on_submit():

        # 提交表单后重定向到index函数
        session['place'] = form.place.data
        session['type'] = form.type.data
        session['log'] = []
        session['matrix_path'] = ""
        # 查询BP中type的数值最高的
        query_BP_by_place = Modelinfo.query.filter(Modelinfo.modelname.like(form.place.data + '_BP_%')).order_by(
            db.desc(form.type.data)).first()
        session['log'].append("query_BP_by_place.modelname:%s" % query_BP_by_place.modelname)
        # # 继续查询绑定查询到的模型的 预测信息
        # query_BPpredict_by_place = Predict.query.filter(Predict.modelname == query_BP_by_place.modelname).order_by(
        #     Predict.datetime).all()
        # # # 查询ELMAN中type的数值最高的
        # # query_ELMAN_by_place = Modelinfo.query.filter(Modelinfo.modelname.like(form.place.data + '_ELMAN_%')).order_by(
        # #     db.desc(form.type.data)).first()
        # # session['log'].append("query_ELMAN_by_place.modelname:%s"%query_ELMAN_by_place.modelname)
        # # query_ELMANpredict_by_place = Predict.query.filter(Predict.modelname == query_BP_by_place.modelname).order_by(
        # #     Predict.datetime).all()
        #
        #
        # # 查询真实人数数据
        # query_true_by_place = Predict.query.filter(Predict.modelname.like(form.place.data + '_true%')).order_by(
        #     Predict.datetime).all()
        #
        # datetime = []
        # TRUE = []
        # BP = []
        # for k in range(len(query_true_by_place)):
        #     datetime.append(query_true_by_place[k].datetime)
        #     TRUE.append(query_true_by_place[k].peoplenum)
        #     BP.append(query_BPpredict_by_place[k].peoplenum)
        session['matrix_path'] = url_for('static',filename="output/%s.csv" % (form.place.data) )

        # 在蓝本中需要url_for用（蓝本名.函数名）所以main.index 简写为.index
        return redirect(url_for('.index'))

    return render_template('show_all_flask.html',
                           form=form,
                           matrix=session.get('matrix_path'))

