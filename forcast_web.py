from flask import Flask, render_template, session, redirect, url_for
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField,SelectField
from wtforms.validators import Required,AnyOf

app = Flask(__name__)
# 使用密钥防止外站攻击
app.config['SECRET_KEY'] = 'hard to guess string'
bootstrap = Bootstrap(app)

# 表单类
class SelectForm(FlaskForm):
    place = SelectField("地点",choices=[("place1", "place1")])
    type = SelectField("评判标准",choices=[("type1", "type1"),("type2", "type2")])

    # place = StringField("地点")
    # type = StringField("评判标准")
    submit = SubmitField("预定")


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


@app.route('/', methods=['GET', 'POST'])
def index():
    form = SelectForm()
    if form.validate_on_submit():
        # 提交表单后重定向到index函数
        session['place'] = form.place.data
        session['type'] = form.type.data
        return redirect(url_for('index'))
    return render_template('show_all_flask.html', form=form, place=session.get('place'),type=session.get('type'))


if __name__ == '__main__':
    app.run()
