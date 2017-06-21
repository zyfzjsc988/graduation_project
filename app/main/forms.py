# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:forms.py

@time:2017/5/22 15:53

@desc:
所有的表单都集中在这里
"""
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField

# page1提交表单
class SelectForm(FlaskForm):
    place = SelectField("地点",choices=[("place1", "place1")])
    # type = SelectField("评判标准",
    #                    choices=[("trainaccuracy", "train_accuracy"),
    #                             ("testaccuracy", "test_accuracy"),
    #                             ("trainloss", "train_loss"),
    #                             ("testloss", "test_loss")])
    submit = SubmitField("提交")


class SelectPlaceForm(FlaskForm):
    place = SelectField("地点", choices=[("place1", "place1")])
    submit = SubmitField("提交")

class SelectForecastForm(FlaskForm):
    hour = SelectField("你想预测几小时后的客流情况：",choices=[("1小时", "1小时"),("2小时","2小时"),("3小时","3小时"),("4小时","4小时"),("5小时","5小时")])
    submit = SubmitField("提交")