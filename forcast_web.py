# -*- coding: utf-8 -*-
from flask import Flask,url_for

app = Flask(__name__)

# route装饰器 过滤URL=’/’可以触发其装饰的函数
# 该函数名在生成URL时被特定的函数采用，该函数返回显示在浏览器中的信息
@app.route('/')
def hello_world():
    return 'Hello World!'
# host:5000/hello 打印出hello JOY!
@app.route('/hello')
def hello():
    return 'Hello JOY!'

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






# 模块名：模块名为MAIN则其作为单独应用启动，否则可以作为某可导入模块
if __name__ == '__main__':
    # 开启debug模式
    app.debug = True
    # 执行服务器 host参数指定可访问的IP。
    # 127.0.0.1为localhost，指定IP=本机IP则在该局域网下的终端都可访问到。0.0.0.0为公网IP
    app.run(host='192.168.1.110')
