## Prerequisites: Account at PhysioNet + link to AWS + 'aws configure'

# Import libraries
from __future__ import print_function

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import dill
import importlib

# Import AWS libraries
from pyathena import connect
from pyathena.util import as_pandas
import boto3
from botocore.client import ClientError

import helper_functions
importlib.reload(helper_functions)
# from helper_functions import table_exists_in_athena, query_to_dataframe

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

# %% Create table that includes the type of ventilation therapy
table_vent_ther = helper_functions.create_table(cursor, 'vent_therapy_type.sql', sql_queries_dir, database_name)

# %% Create table with duration of mechanical ventilation (vent durations are
# important in order to capture the start and end of the segment of the
# self-extubation)
table_vent_dur = helper_functions.create_table(cursor, 'ventilation_durations.sql', sql_queries_dir, database_name)

# %% Create table with demographics (height, weight, BMI)
table_demogr = helper_functions.create_table(cursor, 'height_weight_bmi.sql', sql_queries_dir, database_name)


# Get patients with self-extubation events
data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

# Get total number of self-extubation events
query_get_se_events = """SELECT COUNT(*) FROM {table_vent_ther} WHERE selfextubated = 1""".\
    format(table_vent_ther=table_vent_ther)
df_se_events = helper_functions.query_to_dataframe(cursor, query_get_se_events)
print('Found {:d} self-extubation events!'.format(df_se_events.iloc[0,0]))

table_icustays = database_name + '.icustays'
query_se_cohort_info = """
WITH inter_1 AS
(
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
), inter_2 AS
(
    SELECT inter_1.icustay_id, vent_start, vent_end, se_charttime, duration_hours
    FROM inter_1
    LEFT JOIN {table_demogr} as demogr
    ON inter_1.icustay_id = demogr.icustay_id
    WHERE se_diff = min_se_diff
)
SELECT icu.subject_id, icu.icustay_id, vent_start, vent_end, se_charttime, duration_hours
FROM inter_2
LEFT JOIN {table_icu} as icu
ON inter_2.icustay_id = icu.icustay_id
""".format(table_vent_ther=table_vent_ther, \
    table_dur=table_vent_dur, table_demogr=table_demogr, table_icu=table_icustays)

df_se_cohort = helper_functions.query_to_dataframe(cursor, query_se_cohort_info, \
    df_file_name='se_co_info')
df_se_cohort
df_se_cohort.describe()

sum(df_se_cohort['icustay_id'].value_counts() == 1)
