# -*- coding: utf-8 -*-
from flask import Flask
from flask import url_for
from flask import request
from flask import render_template

app = Flask(__name__)

# route装饰器 过滤URL=’/’可以触发其装饰的函数
# 该函数名在生成URL时被特定的函数采用，该函数返回显示在浏览器中的信息
@app.route('/')
def hello_world():
    return 'Hello World!'


# 使用变量URL ：<variable_name>作为命名参数进行传递
@app.route('/user/<username>')
def show_username(username):
    return 'UserName is %s' % username
# <converter:variable_name>作为转换器:转换器有int\float\path
@app.route('/post/<int:post_id>')
def show_postID(post_id):
    return 'Post ID is %d' % post_id

# 可以使用url_for模块构造url
with app.test_request_context():
    print(url_for('show_username',username='Joy Zhou'))
    # 给css静态文件生成一个url
    print(url_for('static',filename = 'style.css'))

# http method :客户端想对请求的页面做什么
# {GET：获取页面上的信息并发给我}，{POST：想在URL上发布新信息}，{HEAD:获取页面的消息头（与GET类似）}，
# {PUT：类似POST会触发存储过程多次}，{DELETE：删除给定位置的信息}，{OPTIONS：自动处理}
@app.route('/login',methods=['GET','POST','HEAD','PUT','DELETE'])
def login():
    if request.method == 'POST':
        do_the_login()
    else:
        show_the_login_form()
def do_the_login():
    print('do the login')
def show_the_login_form():
    print('show the login form')


# 模板渲染
# 使用render_template() 来渲染模板
# flask 会在templates文件夹里寻找模板hello.html，文件夹必须与该模块同级
@app.route('/hello/')
@app.route('/hello/<name>')
def hello(name = None):
    return render_template('hello.html',name = name)





# 模块名：模块名为MAIN则其作为单独应用启动，否则可以作为某可导入模块
if __name__ == '__main__':
    # 开启debug模式
    app.debug = True
    # 执行服务器 host参数指定可访问的IP。
    # 127.0.0.1为localhost，指定IP=本机IP则在该局域网下的终端都可访问到。0.0.0.0为公网IP
    app.run(host='192.168.1.110')
