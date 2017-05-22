from flask import Flask, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField,SelectField
from flask_sqlalchemy import SQLAlchemy
import logging

app = Flask(__name__)
# 使用密钥防止外站攻击
app.config['SECRET_KEY'] = 'hard to guess string'


# 配置数据库
username = 'root'
password = 'password'
hostname = 'localhost:3306'
dbname = 'flownn'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://%s:%s@%s/%s' % (username,password,hostname,dbname)
app.config['SQLALCHEMY_COMMIT_ON_TEARDOWN'] = True

# 使用bootstrap框架渲染
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)

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

# 访问静态文件
# url = url_for('static',filename = 'output/'+'一系列信息.json',_external=True)

# 表单类
class SelectForm(FlaskForm):

    place = SelectField("地点",choices=[("place1", "place1")])
    type = SelectField("评判标准",choices=[("type1", "type1"),("type2", "type2")])

    submit = SubmitField("提交")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SelectForm()
    query_result = Modelgrade.query.all()
    log = ""
    for item in query_result:
        form.place.choices.append((item.placename,item.placename))
        form.type.choices.append((item.losstype,item.losstype))
    if form.validate_on_submit():
        # 提交表单后重定向到index函数
        session['place'] = form.place.data
        session['type'] = form.type.data

        return redirect(url_for('index'))
    return render_template('show_all_flask.html',log=log, form=form, place=session.get('place'),type=session.get('type'))


if __name__ == '__main__':
    app.run()
