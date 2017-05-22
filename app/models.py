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
class Modelgrade(db.Model):
    __tablename__ = 'modelgrade'
    nnname = db.Column(db.String(45),primary_key=True)
    placename = db.Column(db.String(45),primary_key=True)
    losstype = db.Column(db.String(45),primary_key=True)
    opttype = db.Column(db.String(45),primary_key=True)
    train_loss = db.Column(db.Float)
    train_acc = db.Column(db.Float)
    test_acc = db.Column(db.Float)
    train_loss = db.Column(db.Float)
    activation = db.Column(db.String(45),primary_key=True)
    dropout = db.Column(db.Boolean,primary_key=True)
    # 没有外键也没有表之间的关联关系

    def __repr__(self):
        return '<modelgrade %r_%r>' % (self.nnname,self.losstype)