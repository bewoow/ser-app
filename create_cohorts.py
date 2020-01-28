## Prerequisites: Account at PhysioNet + link to AWS + 'aws configure'

# Import libraries
# from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import os
import re
import dill
import importlib
import wfdb
import requests

import matplotlib
matplotlib.rcParams['figure.dpi'] = 144
import matplotlib.pyplot as plt

# Import AWS libraries
from pyathena import connect
from pyathena.util import as_pandas
import boto3
from botocore.client import ClientError

from helper_functions import create_table, query_to_dataframe, get_chart_data, extract_wfdb_numerics

# Temporary for notebooks
pd.options.display.max_rows = None
import helper_functions
importlib.reload(helper_functions)

s3 = boto3.resource('s3')
client = boto3.client('sts')
account_id = client.get_caller_identity()['Account']
my_session = boto3.session.Session()
region = my_session.region_name

athena_query_results_bucket = 'tdi-project-query-results-' + account_id + '-' + region

try:
    s3.meta.client.head_bucket(Bucket=athena_query_results_bucket)
except ClientError:
    bucket = s3.create_bucket(Bucket=athena_query_results_bucket)
    print('Creating bucket ' + athena_query_results_bucket)

cursor = connect(s3_staging_dir='s3://' + athena_query_results_bucket + '/athena/temp').cursor()

database_name = 'mimiciii'
sql_queries_dir = './sql_queries'

# %% Create table that includes the type (generally) of ventilation therapy. It
# includes timestatmps indicating initation of oxygen therapy, initation of MV,
# of extubation and of self-extubation.
table_vent_ther = create_table(cursor, 'vent_therapy_type.sql', sql_queries_dir, database_name)

# %% Create table with duration of mechanical ventilation. Vent durations are
# important in order to capture the start and end of the MV segment prior to the
# self-extubation event (whose timestamp has been stored in the vent therapy table).
table_vent_dur = create_table(cursor, 'ventilation_durations.sql', sql_queries_dir, database_name)

# %% Create table with demographics (height, weight, BMI)
table_demogr = create_table(cursor, 'height_weight_bmi.sql', sql_queries_dir, database_name)

# %% Get all occurrences of self-extubation
data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

table_icustays = database_name + '.icustays'
query_se_cohort_info = """WITH inter_1 AS
    (
        -- More than a single self-extubation may be charted for the same event;
        -- Compute the minimun difference from the end of ventilation chart time
        SELECT
            vent_ther.icustay_id,
            vent_ther.charttime AS se_charttime,
            vent_dur.starttime AS vent_start,
            vent_dur.endtime AS vent_end,
            vent_dur.duration_hours,
            ABS(date_diff('hour', vent_dur.endtime, vent_ther.charttime)) AS se_diff,
            MIN(ABS(date_diff('hour', vent_dur.endtime, vent_ther.charttime)))
                OVER (PARTITION BY vent_ther.icustay_id) AS min_se_diff
        FROM {table_vent_ther} AS vent_ther
        JOIN {table_dur} AS vent_dur
        ON vent_ther.icustay_id = vent_dur.icustay_id
        WHERE vent_ther.selfextubated = 1
    ),
    -- Keep the one with the smallest difference from the vent end time
    -- Join with demographis table
    inter_2 AS
    (
        SELECT inter_1.icustay_id, vent_start, vent_end, se_charttime, duration_hours,
            demogr.weight, demogr.height, demogr.BMI
        FROM inter_1
        LEFT JOIN {table_demogr} as demogr
        ON inter_1.icustay_id = demogr.icustay_id
        WHERE se_diff = min_se_diff
    )
    -- Get the subject IDs from icu stays table
    SELECT icu.subject_id, icu.hadm_id, inter_2.*
    FROM inter_2
    LEFT JOIN {table_icu} as icu
    ON inter_2.icustay_id = icu.icustay_id""".format(table_vent_ther=table_vent_ther, \
        table_dur=table_vent_dur, table_demogr=table_demogr, table_icu=table_icustays)

df_se_cohort = query_to_dataframe(cursor, query_se_cohort_info, df_file_name='se_co_info')
print('Found {:d} self-extubation events in total!'.format(df_se_cohort.shape[0]))

# %% Cleaning tables to eliminate multiple events of the same patient
# A single patient may have multiple self-extubation events. Keep the one with
# the largest duration.
df_se_cohort_reduced = df_se_cohort.copy()
icustays_multiple_se = df_se_cohort_reduced['icustay_id'].value_counts() > 1 # icustay_id with more than one event

for icustay_id in icustays_multiple_se[icustays_multiple_se == True].index:
    instances = df_se_cohort_reduced[df_se_cohort_reduced['icustay_id'] == icustay_id]

    # Removing table entries (of a single patient) with MV duration less than the maximum
    max_duration = instances['duration_hours'].max()
    for index in instances.index:
        if df_se_cohort_reduced.loc[index, 'duration_hours'] != max_duration:
            df_se_cohort_reduced = df_se_cohort_reduced.drop(index)

    # Removing possible entries associated with the same SE event (selecting the
    # entry with SE charted time closer to the end of MV)
    instances = df_se_cohort_reduced[df_se_cohort_reduced['icustay_id'] == icustay_id]
    if instances.shape[0] > 1: # not necessary, but included to make sure that a single entry is not deleted
        instances = instances.assign(end_mv_se_diff=instances['se_charttime'] - instances['vent_end'])

        # Add NaN if difference is negative (will be removed)
        instances.loc[instances['end_mv_se_diff'] < pd.Timedelta(0), 'end_mv_se_diff'] = np.nan
        instances = instances.sort_values(by=['end_mv_se_diff'])
        # remove all but the first (entry with smallest differencec)
        for i in instances.index[1:]:
            df_se_cohort_reduced = df_se_cohort_reduced.drop(i)
print('Found {:d} distinct self-extubation events (different patients)!'.format(df_se_cohort_reduced.shape[0]))

# %% Final SE cohort with duration of MV > 12 hours
df_se_final_cohort = df_se_cohort_reduced[df_se_cohort_reduced['duration_hours'] >= 12]

# plt.hist(df_se_final_cohort['duration_hours'].dropna(), 20)

chart_data = dict()
for icustay_id in df_se_final_cohort['icustay_id'].values[:10]:
    mv_start_time = df_se_final_cohort[df_se_final_cohort.icustay_id == icustay_id]['vent_start'].iloc[0]
    mv_end_time = df_se_final_cohort[df_se_final_cohort.icustay_id == icustay_id]['vent_end'].iloc[0]
    subject_id = df_se_final_cohort[df_se_final_cohort.icustay_id == icustay_id]['subject_id'].iloc[0]
    chart_data[icustay_id] = get_chart_data(cursor, subject_id, icustay_id, mv_start_time, mv_end_time)

chart_data.keys()
# Get numerics from MIMIC-III Waveform Database (matched with MIMIC-III Clinical Database)
# matched_record_list = wfdb.io.get_record_list('mimic3wdb/matched')
html = requests.get('https://archive.physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-numerics')
numerics_record_list = html.text

numerics = dict()
for icustay_id in df_se_final_cohort['icustay_id'].values:
    subject_id = df_se_final_cohort[df_se_final_cohort.icustay_id == icustay_id]['subject_id'].iloc[0]
    numerics[icustay_id] = extract_wfdb_numerics(subject_id, numerics_record_list)
numerics

icustay_id = list(numerics.keys())[1]

plt.plot(numerics[icustay_id]['time'], numerics[icustay_id]['HR'])
plt.plot(se_charttime, 20, '*')
plt.plot

# icustay_id = 235261 -> 491


# df_test[df_test['vital_sign'] == 'GCStot'].sort_values('charttime')
#
# plt.plot(df_test[df_test['vital_sign'] == 'GCStot'].loc[:,['charttime']], df_test[df_test['vital_sign'] == 'GCStot'].loc[:,['valuenum']], '.')
#
df_test = chart_data[icustay_id]
df_1 = df_test[df_test['vital_sign'] == 'RR'].sort_values(by='charttime')
df_4 = df_test[df_test['vital_sign'] == 'RR'].groupby(['charttime']).mean()
# df_2 = df_test[df_test['vital_sign'] == 'MBP'].sort_values(by='charttime')
# df_3 = df_test[df_test['vital_sign'] == 'SBP'].groupby(['charttime']).mean()
#
# plt.plot(df_2['charttime'], df_2['valuenum'], '*')
# plt.plot(df_3.index, df_3['valuenum'], '.r')
# plt.plot(df_1['charttime'], df_1['valuenum'], 'xg')
plt.plot(df_4.index, df_4['valuenum'], '.r')
plt.plot(df_numerics['time'], df_numerics['RESP'])
plt.plot(se_charttime, 20, '*')
