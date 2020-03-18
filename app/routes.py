# TDI Capstone Project – Self-extubation Risk (SeR) Score

from flask import render_template, request, redirect

from app import app
from app.lib import get_se_cohort, get_chart_data, create_bokeh_viz_chartdata, \
    predict_ser_risk, create_bokeh_viz_ser_risk

import numpy as np
from functools import lru_cache

df_se_cohort_reduced = get_se_cohort()
# Final SE cohort with duration of MV > 12 hours
MV_HOURS = 12
se_cohort = df_se_cohort_reduced[df_se_cohort_reduced['duration_hours'] >= MV_HOURS]


@app.route('/', methods=['GET', 'POST'])
def home():
    icustay_ids = [str(id) for id in list(se_cohort['icustay_id'])]
    icustay_ids.sort()

    if request.method == 'GET':
        return render_template('home.html', icustay_ids=icustay_ids, num_se=se_cohort.shape[0], mv_hours=MV_HOURS,
                               sel_icustay_id='', sel_subject_id='', sel_adm_id='', sel_weight='', sel_height='',
                               sel_bmi='', sel_start_mv='', sel_end_mv='', sel_dur_mv='')
    else:
        if request.form.get('icustay_id_selection'):
            sel_icustay_id = request.form.get('icustay_id_selection')
            sel_subject_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['subject_id'].iloc[0])
            sel_adm_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['hadm_id'].iloc[0])
            sel_weight = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['weight'].iloc[0])
            sel_height = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['height'].iloc[0])
            sel_bmi = str(np.round(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['bmi'].iloc[0], 2))
            sel_start_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_start'].iloc[0])
            sel_end_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_end'].iloc[0])
            sel_dur_mv = str(np.round(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['duration_hours'].iloc[0], 2))

            return render_template('home.html', icustay_ids=icustay_ids, num_se=se_cohort.shape[0], mv_hours=MV_HOURS,
                                   sel_icustay_id=sel_icustay_id, sel_subject_id=sel_subject_id, sel_adm_id=sel_adm_id,
                                   sel_weight=sel_weight, sel_height=sel_height, sel_bmi=sel_bmi, sel_start_mv=sel_start_mv,
                                   sel_end_mv=sel_end_mv, sel_dur_mv=sel_dur_mv)
        else:
            return render_template('home.html', icustay_ids=icustay_ids, num_se=df_se_cohort_reduced.shape[0],
                                   sel_icustay_id='', sel_subject_id='', sel_adm_id='', sel_weight='', sel_height='',
                                   sel_bmi='', sel_start_mv='', sel_end_mv='', sel_dur_mv='')


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
    df_ser_pred = get_ser_risk_to_plot(sel_icustay_id)

    script_chartdata, div_chartdata = create_bokeh_viz_chartdata(chart_data, sel_vital_sign)
    if df_ser_pred is not None:
        script_ser, div_ser = create_bokeh_viz_ser_risk(df_ser_pred)
    else:
        script_ser = None
        div_ser = None

    return render_template('plot.html', sel_icustay_id=sel_icustay_id, vital_signs=vital_signs,
                           sel_vs=sel_vital_sign, script_chartdata=script_chartdata, div_chartdata=div_chartdata,
                           script_ser=script_ser, div_ser=div_ser)


@lru_cache()
def get_chart_data_to_plot(sel_icustay_id):
    sel_subject_id = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['subject_id'].iloc[0])
    sel_start_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_start'].iloc[0])
    sel_end_mv = str(se_cohort[se_cohort['icustay_id'] == int(sel_icustay_id)]['vent_end'].iloc[0])

    chart_data = get_chart_data(sel_subject_id, sel_icustay_id, sel_start_mv, sel_end_mv)
    vital_signs = list(chart_data['vital_sign'].dropna().unique())
    vital_signs.sort()

    return chart_data, vital_signs

@lru_cache()
def get_ser_risk_to_plot(sel_icustay_id):
    # Get chart data
    chart_data, _ = get_chart_data_to_plot(sel_icustay_id)

    # Get SER prediction.
    df_ser_pred = predict_ser_risk(chart_data)

    # Reorganize dataframe to add another entry for plotting purposes (ZOH style).
    df_ser_pred_exp = df_ser_pred.copy()
    for i in df_ser_pred.index:
        temp = df_ser_pred.loc[i].copy()
        temp['seg_end_time'] = temp['seg_start_time'] + np.timedelta64(1, 's')
        df_ser_pred_exp = df_ser_pred_exp.append(temp)

    df_ser_pred_exp.rename(columns={'seg_end_time': 'time'}, inplace=True)
    del df_ser_pred_exp['seg_start_time']

    df_ser_pred_exp = df_ser_pred_exp.set_index('time')
    df_ser_pred_exp.sort_index(inplace=True)

    return df_ser_pred_exp

if __name__ == '__main__':
    app.run()
