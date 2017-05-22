import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    """
    配置类：基类
    """
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard to guess string'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 配置数据库
    username = 'root'
    password = 'password'
    hostname = 'localhost:3306'

    @staticmethod
    def init_app(app):
        """
        对当前环境的配置初始化
        :param app: 
        :return: 
        """
        pass


class DevelopmentConfig(Config):
    """
    开发环境配置
    """
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (Config.username, Config.password, Config.hostname, 'flownn')


class TestingConfig(Config):
    """
    测试环境配置
    """
    TESTING = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('TEST_DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (Config.username, Config.password, Config.hostname, 'flownn-test')


class ProductionConfig(Config):
    """
    生产环境配置
    """
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'mysql://%s:%s@%s/%s' % (Config.username, Config.password, Config.hostname, 'flownn')

# config字典
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
