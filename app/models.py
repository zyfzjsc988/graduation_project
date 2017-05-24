# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:model.py

@time:2017/5/22 16:22

@desc:
数据库中实例model
"""
from . import db

# 定义模型与数据库的表对应
class Place(db.Model):
    __tablename__='place'
    id =db.Column(db.String(4),primary_key=True)
    placename = db.Column(db.String(45))

class Modelinfo(db.Model):
    __tablename__='modelinfo'
#     表结构

    id = db.Column(db.Integer, primary_key=True)
    modelname = db.Column(db.String(256),
                          db.ForeignKey('modelhistory.modelname',ondelete='CASCADE', onupdate='CASCADE'))
    trainloss = db.Column(db.Float)
    trainaccuracy = db.Column(db.Float)
    testloss = db.Column(db.Float)
    testaccuracy = db.Column(db.Float)

class Modelhistory(db.Model):
    __tablename__ = 'modelhistory'
    #     表结构

    id = db.Column(db.Integer, primary_key=True)
    modelname = db.Column(db.String(256))
    epoch = db.Column(db.Integer)
    loss = db.Column(db.Float)
    accuracy = db.Column(db.Float)

    #   一对多关系
    models = db.relationship('Modelinfo')

