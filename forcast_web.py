# -*- coding: utf-8 -*-
from flask import Flask

app = Flask(__name__)

# route装饰器 过滤URL=’/’可以触发其装饰的函数
# 该函数名在生成URL时被特定的函数采用，该函数返回显示在浏览器中的信息
@app.route('/')
def hello_world():
    return 'Hello World!'

# 模块名：模块名为MAIN则其作为单独应用启动，否则可以作为某可导入模块
if __name__ == '__main__':
    # 开启debug模式
    app.debug = True
    # 执行服务器 host参数指定可访问的IP。
    # 127.0.0.1为localhost，指定IP=本机IP则在该局域网下的终端都可访问到。0.0.0.0为公网IP
    app.run(host='192.168.1.110')