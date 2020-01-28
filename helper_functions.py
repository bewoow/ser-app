import os
import glob
import re
import pandas as pd
import requests
from pyathena.util import as_pandas
from datetime import datetime

import wfdb

def create_table(cursor, query_file_name, queries_dir='./sql_queries', database_name='mimiciii', del_table=False):
    """Check, create, and optionally delete table in AWS.
    """
    query_file_name = re.sub(r'[.]\w*', '', query_file_name) + '.sql'

    f = os.path.join(queries_dir, query_file_name)
    with open(f) as fp:
        query = ''.join(fp.readlines())

    table_name = re.search(r'CREATE TABLE ([A-Z]*[.][\w]*)', query).group(1).\
        replace('DATABASE', database_name)

    if del_table:
        print('Deleting table "{}" in AWS...'.format(table_name))
        cursor.execute('DROP TABLE IF EXISTS {};'.format(table_name))
        print('-> Done!')

    if not table_exists_in_aws(cursor, table_name):
        print('Creating table "{}" in AWS...'.format(table_name))
        cursor.execute(query.replace('DATABASE', database_name))
        print('-> Done!')

    return table_name

def table_exists_in_aws(cursor, table_name):
    """Check if table exists in AWS S3."""
    try:
        cursor.execute("SELECT * FROM {} LIMIT 0;".format(table_name))
        print('Table "{}" exists in AWS. OK!'.format(table_name))
        return True
    except Exception:
        print('Table "{}" does NOT exist in AWS.'.format(table_name))
        return False

def query_to_dataframe(cursor, query, df_file_name='', data_dir='./data'):
    """Check presence of stored dataframe in the disk. If not, run query and
    save data"""
    if df_file_name is '':
        print('Dataframe is being collected without saving...')
        return as_pandas(cursor.execute(query))

    # Create directory/ries if it does not exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    df_file_name = re.sub(r'[.]\w*', '', df_file_name) + '.h5'
    dataframe_path = os.path.join(data_dir, df_file_name)

    if not os.path.exists(dataframe_path):
        print('Dataframe "{}" does not exist in disk. Re-quering and saving it...'.format(df_file_name))

        df = as_pandas(cursor.execute(query))
        df.to_hdf(dataframe_path, key='df', mode='w')
        print('-> Done!')

        return df
    else:
        print('Dataframe "{}" exists in disk. Loading it...'.format(df_file_name))
        return pd.read_hdf(dataframe_path, 'df')

def get_chart_data(cursor, subject_id, icustay_id, start_time, end_time, queries_dir='./sql_queries', database_name='mimiciii'):

    f = os.path.join(queries_dir, 'vitals.sql')
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
