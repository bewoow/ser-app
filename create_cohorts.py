## Prerequisites: Account at PhysioNet + link to AWS + 'aws configure'

# Import libraries
import datetime
import numpy as np
import pandas as pd
import os
import re
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

from helper_functions import create_table, query_to_dataframe

# Extract info for AWS connection
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

data_dir = './data'
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

#TODO: we don't need to verify tables in AWS if all data have been downloaded
def get_all_se_occurences():
    # Create table that includes the type (generally) of ventilation therapy. It
    # includes timestatmps indicating initation of oxygen therapy, initation of MV,
    # of extubation and of self-extubation.
    table_vent_ther = create_table(cursor, 'vent_therapy_type.sql', sql_queries_dir, database_name)

    # Create table with duration of mechanical ventilation. Vent durations are
    # important in order to capture the start and end of the MV segment prior to the
    # self-extubation event (whose timestamp has been stored in the vent therapy table).
    table_vent_dur = create_table(cursor, 'ventilation_durations.sql', sql_queries_dir, database_name)

    # Create table with demographics (height, weight, BMI)
    table_demogr = create_table(cursor, 'height_weight_bmi.sql', sql_queries_dir, database_name)

    # Get all occurrences of self-extubation
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

    df_se_all_occurences = query_to_dataframe(cursor, query_se_cohort_info, df_file_name='se_co_info')
    print('Found {:d} self-extubation events in total!'.format(df_se_all_occurences.shape[0]))

    return df_se_all_occurences

def get_se_cohort():
    df_se_all_occurences = get_all_se_occurences()

    # Cleaning tables to eliminate multiple events of the same patient
    # A single patient may have multiple self-extubation events. Keep the one with
    # the largest duration.
    df_se_cohort_reduced = df_se_all_occurences.copy()
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

    return df_se_cohort_reduced

def get_chart_data(subject_id, icustay_id, start_time, end_time, queries_dir='./sql_queries'):

    f = os.path.join(queries_dir, 'chart_data.sql')
    with open(f) as fp:
        chart_data_query = ''.join(fp.readlines())

    chart_data_query = chart_data_query.replace('DATABASE', database_name)
    chart_data_query = chart_data_query.replace('STARTTIME', str(start_time))
    chart_data_query = chart_data_query.replace('ENDTIME', str(end_time))
    chart_data_query = chart_data_query.replace('ICUSTAY_ID', str(icustay_id))

    local_dl_dir = './data/p' + str(subject_id).zfill(6) + '/'
    file_name = 'p' + str(subject_id).zfill(6) + '_icuid' + str(icustay_id).zfill(6) + '_chartdata'

    return query_to_dataframe(cursor, chart_data_query, df_file_name=file_name, data_dir=local_dl_dir)


def extract_wfdb_numerics(subject_id, numerics_record_list='', data_dir='./data', channel_names=''):
    """Download numerics from MIMIC-III Waveform Database Matched Subset and
    generate a Pandas dataframe.
    """
    # If record list is not provided, download it again.
    if numerics_record_list == '':
        html = requests.get('https://archive.physionet.org/physiobank/database/mimic3wdb/matched/RECORDS-numerics')
        numerics_record_list = html.text

    # Append zeros at the beginning as per the MIMIC naming convention.
    subject_id = str(subject_id).zfill(6)
    # Create directory path on the server with subject's files.
    server_path = 'p' + subject_id[:2] + '/p' + subject_id + '/'

    # Create path of local directory where files will be stored.
    local_dl_dir = data_dir + '/p' + subject_id + '/'

    # Find all numerics records of subject with "subject_id".
    string_to_match = r'\np\d\d/p' + subject_id + r'/(p' + subject_id + \
        r'-\d\d\d\d-\d\d-\d\d-\d\d-\d\dn)'
    subject_numerics = re.findall(string_to_match, numerics_record_list)

    if subject_numerics:
        print('Subject {0} found in MIMIC-III Waveform Database Matched Subset! :)'.format(subject_id))
        # In case not all found numerics records have been download, we re-download all.
        if not sorted([local_dl_dir + elem + '.hea' for elem in subject_numerics]) \
            == sorted(glob.glob(local_dl_dir + '*.hea')):
            # Download files to local directory
            wfdb.io.dl_database(db_dir='mimic3wdb/matched/{0}'.format(server_path), \
                dl_dir=local_dl_dir, records=subject_numerics)
        else:
            print('-> Files for subject {0} have already been downloaded in {1}'.format(subject_id, local_dl_dir))

        df_list = []
        for numerics_filepath in sorted(glob.glob(local_dl_dir + '*.hea')): # sorted glob.glob
            # Extract numerics of pre-determined vital signs, if not provided as input.
            if channel_names == '':
                channel_names = ['HR', 'ABPSys', 'ABPDias', 'ABPMean', 'PAPSys', 'PAPDias', 'PAPMean', \
                    'RESP', 'PULSE', 'SpO2', '%SpO2', 'NBP SYS', 'NBP Sys', 'NBP DIAS', 'NBP Dias', 'NBP MEAN', 'NBP Mean']

            # Get signals and fields from numerics file based on selected channels.
            signals, fields = wfdb.rdsamp(numerics_filepath.replace('.hea', ''), channel_names=channel_names)

            # Create a pandas DateTimeIndex based on numerics start date and time.
            numerics_starttime = datetime.combine(fields['base_date'], fields['base_time'])
            sampl_period = round(1 / fields['fs'])
            numerics_times = pd.date_range(numerics_starttime, periods=fields['sig_len'], freq='{}S'.format(sampl_period))

            # Create dataframe
            df_numerics = pd.DataFrame(data=signals, columns=fields['sig_name'])
            df_numerics['time'] = numerics_times

            # Rename data fram columns (key:old value, value:new value)
            rename_column_dict = {'ABPSys': 'SBP', 'ABPDias': 'DBP', 'ABPMean': 'MBP', \
                'NBP SYS': 'SBP',  'NBP Sys': 'SBP', 'NBP DIAS': 'DBP', 'NBP Dias': 'DBP', \
                'NBP MEAN': 'MBP', 'NBP Mean': 'MBP','RESP': 'RR', '%SpO2': 'SpO2'}
            df_numerics = df_numerics.rename(rename_column_dict, axis=1)
            df_list.append(df_numerics)

        # Concatenate and return dataframe.
        df_numerics_all = pd.concat(df_list, sort=False)
        return df_numerics_all
    else:
        print('Subject {0} NOT found in MIMIC-III Waveform Database Matched Subset :('.format(subject_id))
