import os  # TODO: Change to install modules
import sys
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from functools import lru_cache
from datetime import datetime, timedelta
import joblib

# sys.path.append(os.path.abspath('../app'))

os.chdir('../app/')
from lib import get_se_cohort, get_chart_data, create_bokeh_viz_chartdata, \
    predict_ser_risk, impute_values, compute_total_gcs, compute_rsbi_ve, compute_ser_features, get_ser_analysis_df
os.chdir('../')

# %%
df_se_cohort_reduced = get_se_cohort()
# Final SE cohort with duration of MV > 12 hours
mv_hours = 12
se_cohort = df_se_cohort_reduced[df_se_cohort_reduced['duration_hours'] >= mv_hours]

# Select chartdata from single icustay_id
icustay_id = se_cohort['icustay_id'].values[0]
mv_start_time = se_cohort[se_cohort.icustay_id ==
                          icustay_id]['vent_start'].iloc[0]
mv_end_time = se_cohort[se_cohort.icustay_id == icustay_id]['vent_end'].iloc[0]
subject_id = se_cohort[se_cohort.icustay_id ==
                       icustay_id]['subject_id'].iloc[0]

chart_data = get_chart_data(
    subject_id, icustay_id, mv_start_time, mv_end_time)


seg_size_hours = 3
seg_nptimedelta = np.timedelta64(seg_size_hours, 'h')
# Functions that will be evaluated for each segment to get the feature vector.
functions = ['mean', 'median', 'mode', 'kurtosis', 'skew', 'std']

df_se_analysis = pd.DataFrame()

i = 0
# Get chart data for each icustay, impute values, and compute GCS, RSBI, and VE
chart_data = impute_values(chart_data)
chart_data = compute_total_gcs(chart_data)
chart_data = compute_rsbi_ve(chart_data)

# Extract vital sign
vital_signs = chart_data['vital_sign'].unique()

label_idx = 0  # label '1' close to event, increasing as we get further

# We find the unique charttimes and sort them so that we start from the
# last (when self extubation occurs) and when go back segment-by-segment
sorted_charttimes = sorted(chart_data['charttime'].unique())

seg_end_time = sorted_charttimes[-1]
while seg_end_time > sorted_charttimes[0]:
    seg_start_time = seg_end_time - seg_nptimedelta
    label_idx += 1  # label '1' close to event, increasing as we get further

    # For every vital sign, we extract the data within each segment
    # and compute the different functions (e.g., mean, variance) which
    # will consist our feature vector
    for vital_sign in vital_signs:
        df = chart_data[chart_data['vital_sign'] \
            == vital_sign].sort_values(by='charttime')

        seg_start_idx = df['charttime'].apply(
            lambda x: abs(x - seg_start_time)).idxmin()
        seg_end_idx = df['charttime'].apply(
            lambda x: abs(x - seg_end_time)).idxmin()

        df_se_analysis.loc[i, 'label'] = label_idx
        df_se_analysis.loc[i, 'icustay_id'] = int(df['icustay_id'].iloc[0])
        df_se_analysis.loc[i, 'seg_start_time'] = seg_start_time
        df_se_analysis.loc[i, 'seg_end_time'] = seg_end_time
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

# %%
df_se_data_population_mean = pd.read_hdf(
    './data/pop_mean_seg-size-hours{0}.h5'.format(seg_size_hours))

# Group by label/segment to find how many data we have per label
segment_groups = df_se_analysis.groupby('label')

total_num_hours = 12
num_segments = total_num_hours // seg_size_hours
df_se_data_trun = pd.concat([df_se_analysis.loc[group_idx]
                             for i, group_idx in segment_groups.groups.items() if i <= num_segments])

# Use only the features in the training dataset
columns_to_keep = ['label', 'icustay_id', 'seg_start_time', 'seg_end_time']
columns_to_keep.extend(list(df_se_data_population_mean.index.values))
df_se_data_trun = df_se_data_trun[columns_to_keep]

# Remove features that do not have a lot of entries (i.e., NaNs) for each segment.
segment_groups_trun = df_se_data_trun.groupby('label')

feature_count_seg = segment_groups_trun.count()

for feat in feature_count_seg.columns:
    if feat not in ('icustay_id', 'seg_start_time', 'seg_end_time'):
        # Impute nan feature values with their averages
        for label, label_idx in segment_groups_trun.groups.items():
                df_se_data_trun.loc[label_idx, feat] = df_se_data_trun.loc[label_idx, feat].fillna(
                    df_se_data_population_mean[feat])

#%% Load model and predict
model_filename = './data/gbt_model_eval-hours{0}_seg-size-hours{1}.sav'.format(
    total_num_hours, seg_size_hours)
model = joblib.load(model_filename)

X = df_se_data_trun.drop(['label', 'icustay_id'], axis=1)
y = df_se_data_trun['label']

print(model.predict(X))
print(y)
