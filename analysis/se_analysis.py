import os  # TODO: Change to install modules
import sys
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import lru_cache
from datetime import datetime, timedelta

# sys.path.append(os.path.abspath('../app'))

os.chdir('../app/')
from lib import get_se_cohort, get_chart_data, create_bokeh_viz
os.chdir('../')

df_se_cohort_reduced = get_se_cohort()
# Final SE cohort with duration of MV > 12 hours
mv_hours = 12
se_cohort = df_se_cohort_reduced[df_se_cohort_reduced['duration_hours'] >= mv_hours]

chart_data = dict()
for icustay_id in se_cohort['icustay_id'].values:
    mv_start_time = se_cohort[se_cohort.icustay_id == icustay_id]['vent_start'].iloc[0]
    mv_end_time = se_cohort[se_cohort.icustay_id == icustay_id]['vent_end'].iloc[0]
    subject_id = se_cohort[se_cohort.icustay_id == icustay_id]['subject_id'].iloc[0]
    chart_data[icustay_id] = get_chart_data(subject_id, icustay_id, mv_start_time, mv_end_time)

def impute_values(patient_data):
    """Impute chart daata

    Arguments:
        patient_data {patient_data} -- [description]

    Returns:
        patient_data_imputed -- Imputed chart data
    """
    patient_data = patient_data.dropna(subset=['vital_sign'])

    # Get vital signs available in patient data.
    vital_signs = patient_data['vital_sign'].unique()

    # Extract the unique charttimes that will be imputed if not present.
    charttimes = patient_data['charttime'].unique()

    # For all vital signs, check charttimes and interpolate if missing.
    patient_data_imputed = pd.DataFrame(columns=patient_data.columns)
    for vs in vital_signs:
        df = patient_data[patient_data['vital_sign'] == vs]

        # Take the average value of entries with the same charttime. We use the
        # 'max' for all other entries (e.g., units, etc) so that the aggregated
        # dataframe includes all entries.
        group_by_agg = {column: 'max' for column in df.columns if column != 'charttime'}
        group_by_agg['valuenum'] = 'mean'
        df = df.groupby('charttime', as_index=False).agg(group_by_agg)

        # If a charttime from the charttimes of the vital sign with most
        # entries does not exist, create new row in the dataframe
        # for charttime in charttimes:
        df_2 = pd.DataFrame(columns=patient_data.columns)
        df_2['charttime'] = list(charttimes)
        df = df.append(df_2).drop_duplicates(['charttime'])

        # Apply linear interpolation on the added charttimes. Note that
        # 'icustay_id', 'subject_id', 'unit', and 'vital_sign' should all the
        # same. However, 'unit' and 'vital_sign' are categorical data that get
        # NaN from the interpolation and hence we need to fill them.
        df = df.sort_values(by='charttime').\
            interpolate(method='linear', limit_direction='both', axis=0).\
            fillna(method='ffill').fillna(method='bfill')
        # Convert 'icustay_id', 'subject_id' back to integer values.
        df['icustay_id'] = df['icustay_id'].apply(int)
        df['subject_id'] = df['subject_id'].apply(int)

        patient_data_imputed = patient_data_imputed.append(df, sort=True, ignore_index=True)

    return patient_data_imputed.dropna(subset=['icustay_id'])

def compute_total_gcs(patient_data):
    """Computes the total Glasgow Coma Scale (GCS) score as the sum of the
    individual components (eye, motor, verbal).

    Arguments:
        patient_data {pd.DataFrame} -- Patient data in a Pandas dataframe.

    Returns:
        pd.DataFrame -- Patient data dataframe with total GCS (GCStot) included.
    """
    # Get vital signs available in patient data.
    vital_signs = patient_data['vital_sign'].unique()

    # We verify that all the GCS individual components are part of the data.
    if all([gcs_comp in vital_signs for gcs_comp in ('GCSmotor', 'GCSeye', 'GCSverbal')]):
        # Extract individual GCS component scores.
        GCSmotor = patient_data[patient_data['vital_sign'] == 'GCSmotor'].set_index('charttime')
        GCSeye = patient_data[patient_data['vital_sign'] == 'GCSeye'].set_index('charttime')
        GCSverbal = patient_data[patient_data['vital_sign']
                                 == 'GCSverbal'].set_index('charttime')

        # Create a new dataframe and update 'vital_sign' and 'valuenum' columns.
        df_gcs = pd.DataFrame(columns=GCSmotor.columns)

        df_gcs['valuenum'] = (GCSmotor['valuenum'] + GCSeye['valuenum'] + \
            GCSverbal['valuenum'])
        df_gcs = df_gcs.dropna(subset=['valuenum'])
        df_gcs['vital_sign'] = 'GCStot'
        df_gcs['unit'] = None
        df_gcs['subject_id'] = GCSmotor['subject_id']
        df_gcs['icustay_id'] = GCSmotor['icustay_id']

        # Append to patient_data dataframe.
        patient_data = patient_data.append(df_gcs.reset_index(), sort=True, ignore_index=True)

    # If total GCS exists in patient data, just return
    return patient_data

def compute_rsbi_ve(patient_data):
    """Compute and append RSBI (rapid shallow breathing index) and VE (minute
    ventilation) on the 'patient_data' dataframe

    Arguments:
        patient_data {pd.DataFrame} -- patient data

    Returns:
        pd.DataFrame -- patient_data
    """
    vital_signs = patient_data['vital_sign'].unique()
    def create_rsbi_ve_dfs(vt_name):
        """Create dataframes for RSBI and VE based on provided tidal volume
        name (VTobs, VTspot)

        Arguments:
            vt_name {string} -- Name of VT dataframe to be used (VTspot, VTobs)

        Returns:
            pd.DataFrame -- patient_data that includes RSBI and VE
        """
        RR = patient_data[patient_data['vital_sign']
                          == 'RR'].set_index('charttime')
        VTspot = patient_data[patient_data['vital_sign']
                          == 'VTspot'].set_index('charttime')
        VTobs = patient_data[patient_data['vital_sign']
                         == 'VTobs'].set_index('charttime')

        rsbi = RR.copy()
        ve = RR.copy()

        # Get appropriate tidal volume (obs or spot)
        VT = eval(vt_name + "['valuenum'] / 1000")

        rsbi['valuenum'] = RR['valuenum'] / VT
        rsbi['unit'] = 'bpm/l'
        rsbi['vital_sign'] = 'RSBI' + vt_name.split('VT')[1]

        ve['valuenum'] = RR['valuenum'] * VT
        ve['unit'] = 'min/l'
        ve['vital_sign'] = 'VE' + vt_name.split('VT')[1]

        # Reset indices and append to 'patient_data' dataframe.
        rsbi = rsbi.reset_index()
        pat_dat = patient_data.append(rsbi, sort=True, ignore_index=True)

        ve = ve.reset_index()
        pat_dat = pat_dat.append(ve, sort=True, ignore_index=True)

        return pat_dat

    # Compute and append RSBI and Ve for observed and spontaneous tidal volumes.
    if 'VTobs' in vital_signs:
        patient_data = create_rsbi_ve_dfs('VTobs')
    if 'VTspot' in vital_signs:
        patient_data = create_rsbi_ve_dfs('VTspot')

    return patient_data


seg_size_hours = 3
seg_nptimedelta = np.timedelta64(seg_size_hours, 'h')
functions = ['mean', 'median', 'mode', 'kurtosis', 'skew', 'std']

se_analysis_file = './data/se_analysis_{0}hours.h5'.format(seg_size_hours)
if os.path.exists(se_analysis_file):
    df_se_analysis = pd.read_hdf(se_analysis_file)
else:
    df_se_analysis = pd.DataFrame()

    i = 0
    for idx, icustay in enumerate(list(chart_data.keys())):
        print('Processing ID {0} : {1:d}/{2:d}'.format(icustay, idx+1, len(chart_data.keys())))
        # Get chart data for each icustay, impute values, and compute GCS, RSBI, and VE
        patient_data = chart_data[icustay]
        patient_data = impute_values(patient_data)
        patient_data = compute_total_gcs(patient_data)
        patient_data = compute_rsbi_ve(patient_data)

        # Extract vital sign
        vital_signs = patient_data['vital_sign'].unique()

        label_idx = 0  # label '1' close to event, increasing as we get further

        sorted_charttimes = sorted(patient_data['charttime'].unique())

        seg_end_time = sorted_charttimes[-1]
        while seg_end_time > sorted_charttimes[0]:
            seg_start_time = seg_end_time - seg_nptimedelta
            label_idx += 1

            for vital_sign in vital_signs:
                df = patient_data[patient_data['vital_sign'] \
                    == vital_sign].sort_values(by='charttime')

                seg_start_idx = df['charttime'].apply(
                    lambda x: abs(x - seg_start_time)).idxmin()
                seg_end_idx = df['charttime'].apply(
                    lambda x: abs(x - seg_end_time)).idxmin()

                df_se_analysis.loc[i, 'label'] = label_idx
                df_se_analysis.loc[i, 'icustay_id'] = int(df['icustay_id'].iloc[0])
                for func in functions:
                    if func == 'mode':
                        df_se_analysis.loc[i, '{0}_{1}'.format(vital_sign, func)] = \
                            df.loc[seg_start_idx:seg_end_idx]['valuenum'].apply(func)[
                            0]
                    else:
                        df_se_analysis.loc[i, '{0}_{1}'.format(vital_sign, func)] = \
                            df.loc[seg_start_idx:seg_end_idx]['valuenum'].apply(
                                func)

            seg_end_time = seg_start_time
            i += 1

    df_se_analysis['label'] = df_se_analysis['label'].apply(int)
    df_se_analysis['icustay_id'] = df_se_analysis['icustay_id'].apply(int)

    df_se_analysis.to_hdf(se_analysis_file, 's')

# We use only a portion of the dataset (few entries/patients longer than say 24hours )
# Each group 3 or 4 hours
# TODO: make the grouping adjustable

# group by label to find how many data we have per label
label_groups = df_se_analysis.groupby('label')

# compute mean values for each data and each group -> to be used for imputation
df_se_data_mean = pd.DataFrame({i: label_groups.get_group(i).mean(skipna=True) for i in label_groups.groups.keys()})
df_se_data_mean = df_se_data_mean.drop('icustay_id', axis=0)

df_se_data = pd.concat([df_se_analysis.loc[group_idx]
                        for i, group_idx in label_groups.groups.items() if i <= 8])  # for 3-hour seg (8*3 =24)

# impute nan values

vt_count_group = label_groups.count()
max_entries_group = vt_count_group['icustay_id'].max()
for vt in vt_count_group.columns:
    if all(vt_count_group[vt] < max_entries_group * 2 / 3):
        df_se_data = df_se_data.drop([vt], axis=1)


X = df_se_data.drop(['label', 'icustay_id'], axis=1)
y = df_se_data['label']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import (
    SimpleImputer, KNNImputer, IterativeImputer, MissingIndicator)

rf = RandomForestClassifier()
rf.fit(X, y)
