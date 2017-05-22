# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:manage.py

@time:2017/5/22 16:01

@desc:
用于启动脚本
"""
import os
from app import create_app, db
from app.models import Modelgrade

# 配置app
app = create_app(os.getenv('FLASK_CONFIG') or 'default')

# 启动单元测试
def test():
    """Run the unit tests."""
    import unittest
    tests = unittest.TestLoader().discover('tests')
    unittest.TextTestRunner(verbosity=2).run(tests)




if __name__ == '__main__':
    app.run()
