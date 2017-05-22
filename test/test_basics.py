# -*- coding: utf-8 -*-
"""
@author:J.Zhou

@contact:zyfzjsc988@outlook.com

@file:test_basics.py

@time:2017/5/22 16:37

@desc:
单元测试简易版
"""
import unittest
from flask import current_app
from app import create_app, db


class BasicsTestCase(unittest.TestCase):
    # 在所有测试前执行
    def setUp(self):
        self.app = create_app('testing')
        self.app_context = self.app.app_context()
        self.app_context.push()
        db.create_all()


    # 在所有测试后执行
    def tearDown(self):
        db.session.remove()
        db.drop_all()
        self.app_context.pop()

    # 以test_开头的函数都作为测试函数执行
    def test_app_exists(self):
        self.assertFalse(current_app is None)

    def test_app_is_testing(self):
        self.assertTrue(current_app.config['TESTING'])
