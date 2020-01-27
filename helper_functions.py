import os
import re
import pandas as pd
from pyathena.util import as_pandas

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
        cursor.execute("SELECT * FROM {} LIMIT 1;".format(table_name))
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

def get_chart_data(cursor, icustay_id, start_time, end_time, queries_dir='./sql_queries', database_name='mimiciii'):

    f = os.path.join(queries_dir, 'vitals.sql')
    with open(f) as fp:
        chart_data_query = ''.join(fp.readlines())

    chart_data_query = chart_data_query.replace('DATABASE', database_name)
    chart_data_query = chart_data_query.replace('STARTTIME', start_time)
    chart_data_query = chart_data_query.replace('ENDTIME', end_time)
    chart_data_query = chart_data_query.replace('ICUSTAY_ID', icustay_id)

    return query_to_dataframe(cursor, chart_data_query, df_file_name=icustay_id)
