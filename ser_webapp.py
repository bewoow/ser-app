# TDI Capstone Project – Self-Extubation Risk

from flask import Flask, render_template, request, redirect
from create_cohorts import get_se_cohort, get_chart_data
from bokeh_viz import create_bokeh_viz
import numpy as np
from functools import lru_cache

app = Flask(__name__)

df_se_cohort_reduced = get_se_cohort()
# Final SE cohort with duration of MV > 12 hours
se_cohort = df_se_cohort_reduced[df_se_cohort_reduced['duration_hours'] >= 12]

@app.route('/', methods=['GET', 'POST'])
def home():
    icustay_ids = [str(id) for id in list(se_cohort['icustay_id'])]

    if request.method == 'GET':
        return render_template('home.html', icustay_ids=icustay_ids, num_se=df_se_cohort_reduced.shape[0], \
            sel_icustay_id='', sel_subject_id='', sel_adm_id='', sel_weight='', \
            sel_height='', sel_bmi='', sel_start_mv='', sel_end_mv='', sel_dur_mv='')
    else:
        sel_icustay_id = request.form.get('icustay_id_selection')
        sel_subject_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['subject_id'].iloc[0])
        sel_adm_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['hadm_id'].iloc[0])
        sel_weight = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['weight'].iloc[0])
        sel_height = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['height'].iloc[0])
        sel_bmi = str(np.round(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['bmi'].iloc[0], 2))
        sel_start_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_start'].iloc[0])
        sel_end_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_end'].iloc[0])
        sel_dur_mv = str(np.round(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['duration_hours'].iloc[0], 2))

        return render_template('home.html', icustay_ids=icustay_ids, num_se=df_se_cohort_reduced.shape[0], \
            sel_icustay_id=sel_icustay_id, sel_subject_id=sel_subject_id, sel_adm_id=sel_adm_id, sel_weight=sel_weight, \
            sel_height=sel_height, sel_bmi=sel_bmi, sel_start_mv=sel_start_mv, sel_end_mv=sel_end_mv, sel_dur_mv=sel_dur_mv)

@app.route('/about_me')
def about_me():
    return render_template('about_me.html')

@app.route('/about_se')
def about_se():
    return render_template('about_se.html')

@app.route('/about_mimic')
def about_mimic():
    return render_template('about_mimic.html')

@app.route('/plot', methods=['GET', 'POST'])
def plot():
    if request.method == 'GET':
        sel_vital_sign = "HR"
    else:
        sel_vital_sign = request.form.get("vs")

    sel_icustay_id = request.args.get("sel_id")
    chart_data, vital_signs = get_chart_data_to_plot(sel_icustay_id)

    script, div = create_bokeh_viz(chart_data, sel_vital_sign)

    return render_template('plot.html', sel_icustay_id=sel_icustay_id, vital_signs=vital_signs,\
        sel_vs=sel_vital_sign, script=script, div=div)


@lru_cache()
def get_chart_data_to_plot(sel_icustay_id):
    sel_subject_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['subject_id'].iloc[0])
    sel_start_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_start'].iloc[0])
    sel_end_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_end'].iloc[0])

    chart_data = get_chart_data(sel_subject_id, sel_icustay_id, sel_start_mv, sel_end_mv)
    vital_signs = list(chart_data['vital_sign'].dropna().unique())

    return chart_data, vital_signs

if __name__ == '__main__':
    app.run(debug=True)
