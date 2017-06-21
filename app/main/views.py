# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:views.py

@time:2017/5/22 15:47

@desc:
蓝本中定义的程序业务逻辑路由
"""
from flask import render_template, session, redirect, url_for,make_response
from ..models import Modelhistory,Modelinfo,Place,db
from . import main
from .forms import SelectForm,SelectPlaceForm,SelectForecastForm
import os
history_file_path = "output/history"

#
# @main.route('/<filepath>')
# @main.route('/<path>')
# def today(path):
#     base_dir = os.path.dirname(__file__)
#     resp = make_response(open(os.path.join(base_dir, path)).read())
#     resp.headers["Content-type"]="application/json;charset=UTF-8"
#     return resp
#
# @main.before_app_first_request
# def bf_app_request():
#     session['log'] = []

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
        # session['type'] = form.type.data
        session['matrix_path'] = ""
        # session['BP_name'] = ""
        session['placename'] = ""
        session['matrix_path'] = url_for('static',filename="output/%s.csv" % (form.place.data) )
        # session['BP_name'] = query_BP_by_place.modelname
        session['placename'] = Place.query.filter(Place.id == form.place.data).first().placename
        # 在蓝本中需要url_for用（蓝本名.函数名）所以main.index 简写为.index
        return redirect(url_for('.index'))

    return render_template('show_all_flask.html',
                           form=form,
                           # BP=session.get('BP_name'),
                           place= session.get('placename'),
                           matrix=session.get('matrix_path'))

# 详细信息页
@main.route('/show',methods=['GET','POST'])
def show_info():
    # this page is aimed at compare BP with Elman

    form = SelectPlaceForm()
    form.place.choices = []
    for item in Place.query.all():
        form.place.choices.append((item.id, item.placename))


    if form.validate_on_submit():
        session['place'] = form.place.data
        session['placename'] = Place.query.filter(Place.id == form.place.data).first().placename
        session['pathlist'] =[]
        BP_name_list = []
        BP_list =  Modelinfo.query.filter(Modelinfo.modelname.like(form.place.data + '_BP_%')).all()
        session['json_path'] = url_for('static', filename="output/history_%s.json" % (form.place.data))
        for item in BP_list:
            BP_name_list.append(item.modelname)
        session['pathlist']=BP_name_list
        return redirect(url_for('.show_info'))


    return render_template('show_all.html',
                           form=form,
                           placename=session.get('placename'),
                           json_path = session.get('json_path'),
                           pathlist=session.get('pathlist'))
@main.route('/forecast',methods=['GET','POST'])
def forecast():
    form = SelectPlaceForm()
    form.place.choices = []
    for item in Place.query.all():
        form.place.choices.append((item.id, item.placename))
    fform = SelectForecastForm()
    return render_template('forecast.html'
                           ,form=form,
                           fform = fform
                           )