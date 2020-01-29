# TDI Capstone Project – Self-Extubation Risk

from flask import Flask, render_template, request, redirect
from create_cohorts import get_se_cohort

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    se_cohort = get_se_cohort()
    icustay_ids = [str(id) for id in list(se_cohort['icustay_id'])]

    if request.method == 'GET':
        return render_template('home.html', icustay_ids=icustay_ids, selected_id='')
    else:
        selected_icustay_id = request.form.get('icustay_id_selection')
        return render_template('home.html', icustay_ids=icustay_ids, selected_id=selected_icustay_id)

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

@app.route('/about_se')
def about_se():
    return render_template('about_se.html')

if __name__ == '__main__':
    app.run(debug=True)
