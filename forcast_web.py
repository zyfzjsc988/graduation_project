# -*- coding: utf-8 -*-
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# 配置数据库
username = 'root'
password = 'password'
hostname = 'localhost:3306'
dbname = 'flownn'
app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://%s:%s@%s/%s' % (username,password,hostname,dbname)
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

db = SQLAlchemy(app)

# 定义模型
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

    def __repr__(self):
        return '<modelgrade %r_%r>' % (self.nnname,self.losstype)