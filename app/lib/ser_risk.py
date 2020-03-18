from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import joblib

# Define segment size (in hours) and total number of hours to evaluate for SE risk.
SEG_SIZE_HOURS = 3
TOTAL_NUM_HOURS = 48
# Statistics that will be evaluated for each segment to get the feature vector.
STATS = ['mean', 'median', 'mode', 'kurtosis', 'skew', 'std']

def impute_values(chart_data):
    """Impute chart daata

    Arguments:
        chart_data {pd.DataFrame} -- Patient chart data in a Pandas dataframe.

    Returns:
        chart_data_imputed -- Imputed chart data.
    """
    chart_data = chart_data.dropna(subset=['vital_sign'])

    # Get vital signs available in patient data.
    vital_signs = chart_data['vital_sign'].unique()

    # Extract the unique charttimes that will be imputed if not present.
    charttimes = chart_data['charttime'].unique()

    # For all vital signs, check charttimes and interpolate if missing.
    chart_data_imputed = pd.DataFrame(columns=chart_data.columns)
    for vs in vital_signs:
        df = chart_data[chart_data['vital_sign'] == vs]

        # Take the average value of entries with the same charttime. We use the
        # 'max' for all other entries (e.g., units, etc) so that the aggregated
        # dataframe includes all entries.
        group_by_agg = {
            column: 'max' for column in df.columns if column != 'charttime'}
        group_by_agg['valuenum'] = 'mean'
        df = df.groupby('charttime', as_index=False).agg(group_by_agg)

        # If a charttime from the charttimes of the vital sign with most
        # entries does not exist, create new row in the dataframe
        # for charttime in charttimes:
        df_2 = pd.DataFrame(columns=chart_data.columns)
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

        chart_data_imputed = chart_data_imputed.append(
            df, sort=True, ignore_index=True)

    return chart_data_imputed.dropna(subset=['icustay_id'])


def compute_total_gcs(chart_data):
    """Computes the total Glasgow Coma Scale (GCS) score as the sum of the
    individual components (eye, motor, verbal).

    Arguments:
        chart_data {pd.DataFrame} -- Patient chart data in a Pandas dataframe.

    Returns:
        pd.DataFrame -- Patient data dataframe with total GCS (GCStot) included.
    """
    # Get vital signs available in patient data.
    vital_signs = chart_data['vital_sign'].unique()

    # We verify that all the GCS individual components are part of the data.
    if all([gcs_comp in vital_signs for gcs_comp in ('GCSmotor', 'GCSeye', 'GCSverbal')]):
        # Extract individual GCS component scores.
        GCSmotor = chart_data[chart_data['vital_sign']
                              == 'GCSmotor'].set_index('charttime')
        GCSeye = chart_data[chart_data['vital_sign']
                            == 'GCSeye'].set_index('charttime')
        GCSverbal = chart_data[chart_data['vital_sign']
                               == 'GCSverbal'].set_index('charttime')

        # Create a new dataframe and update 'vital_sign' and 'valuenum' columns.
        df_gcs = pd.DataFrame(columns=GCSmotor.columns)

        df_gcs['valuenum'] = (GCSmotor['valuenum'] + GCSeye['valuenum'] +
                              GCSverbal['valuenum'])
        df_gcs = df_gcs.dropna(subset=['valuenum'])
        df_gcs['vital_sign'] = 'GCStot'
        df_gcs['unit'] = None
        df_gcs['subject_id'] = GCSmotor['subject_id']
        df_gcs['icustay_id'] = GCSmotor['icustay_id']

        # Append to chart_data dataframe.
        chart_data = chart_data.append(
            df_gcs.reset_index(), sort=True, ignore_index=True)

    # If total GCS exists in patient data, just return
    return chart_data


def compute_rsbi_ve(chart_data):
    """Compute and append RSBI (rapid shallow breathing index) and VE (minute
    ventilation) on the 'chart_data' dataframe

    Arguments:
        chart_data {pd.DataFrame} -- Patient chart data in a Pandas dataframe.

    Returns:
        pd.DataFrame -- chart_data
    """
    vital_signs = chart_data['vital_sign'].unique()

    def create_rsbi_ve_dfs(vt_name):
        """Create dataframes for RSBI and VE based on provided tidal volume
        name (VTobs, VTspot)

        Arguments:
            vt_name {string} -- Name of VT dataframe to be used (VTspot, VTobs)

        Returns:
            pd.DataFrame -- chart_data that includes RSBI and VE
        """
        RR = chart_data[chart_data['vital_sign']
                        == 'RR'].set_index('charttime')
        VTspot = chart_data[chart_data['vital_sign']
                            == 'VTspot'].set_index('charttime')
        VTobs = chart_data[chart_data['vital_sign']
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

        # Reset indices and append to 'chart_data' dataframe.
        rsbi = rsbi.reset_index()
        pat_dat = chart_data.append(rsbi, sort=True, ignore_index=True)

        ve = ve.reset_index()
        pat_dat = pat_dat.append(ve, sort=True, ignore_index=True)

        return pat_dat

    # Compute and append RSBI and Ve for observed and spontaneous tidal volumes.
    if 'VTobs' in vital_signs:
        chart_data = create_rsbi_ve_dfs('VTobs')
    if 'VTspot' in vital_signs:
        chart_data = create_rsbi_ve_dfs('VTspot')

    return chart_data


def compute_ser_features(chart_data, seg_size_hours=3, stats=['mean']):
    """Compute features for self-extubation risk prediction. Features are
    computed by applying the desired statistics (e.g., mean, median, etc)
    on the chart data.

    Arguments:
        chart_data {pd.DataFrame} -- Patient chart data in a Pandas dataframe.
        seg_size_hours {int} -- Segment duration (in hours) for statistics
        stats {list} -- List of statistics to evaluate.

    Returns:
        pd.DataFrame -- Features based on the statistics
    """
    df_ser_features = pd.DataFrame()

    # Extract vital signs
    vital_signs = chart_data['vital_sign'].unique()

    label_idx = 0  # label '1' close to event, increasing as we get further

    # We find the unique charttimes and sort them so that we start from the
    # last (when self extubation occurs) and when go back segment-by-segment
    sorted_charttimes = sorted(chart_data['charttime'].unique())

    i = 0
    seg_nptimedelta = np.timedelta64(seg_size_hours, 'h')
    seg_end_time = sorted_charttimes[-1]
    while seg_end_time > sorted_charttimes[0]:
        seg_start_time = seg_end_time - seg_nptimedelta
        label_idx += 1  # label '1' close to event, increasing as we get further

        # For every vital sign, we extract the data within each segment
        # and compute the different statistics (e.g., mean, variance) which
        # will consist our feature vector
        for vital_sign in vital_signs:
            df = chart_data[chart_data['vital_sign']
                            == vital_sign].sort_values(by='charttime')

            seg_start_idx = df['charttime'].apply(
                lambda x: abs(x - seg_start_time)).idxmin()
            seg_end_idx = df['charttime'].apply(
                lambda x: abs(x - seg_end_time)).idxmin()

            df_ser_features.loc[i, 'label'] = label_idx
            df_ser_features.loc[i, 'icustay_id'] = int(df['icustay_id'].iloc[0])
            df_ser_features.loc[i, 'seg_start_time'] = seg_start_time
            df_ser_features.loc[i, 'seg_end_time'] = seg_end_time
            for stat in stats:
                if stat == 'mode':
                    df_ser_features.loc[i, '{0}_{1}'.format(vital_sign, stat)] = \
                    df.loc[seg_start_idx:seg_end_idx]['valuenum'].apply(stat)[0]
                else:
                    df_ser_features.loc[i, '{0}_{1}'.format(vital_sign, stat)] = \
                        df.loc[seg_start_idx:seg_end_idx]['valuenum'].apply(stat)

        seg_end_time = seg_start_time
        i += 1

    df_ser_features['label'] = df_ser_features['label'].apply(int)
    df_ser_features['icustay_id'] = df_ser_features['icustay_id'].apply(int)

    return df_ser_features


def get_ser_analysis_df(chart_data):
    """Create dataframe ready for Self-extubation risk prediction.

    Arguments:
        chart_data {pd.DataFrame} -- Patient chart data in a Pandas dataframe.

    Returns:
        pd.DataFrame -- Dataframe with features after imputation and truncation.
    """
    # Retrieve average values of the features from the entire population.
    pop_mean_filename = './data/pop_mean_seg-size-hours{0}.h5'.format(
        SEG_SIZE_HOURS)
    if os.path.exists(pop_mean_filename):
        ser_feat_population_mean = pd.read_hdf(pop_mean_filename)
    else:
        return None

    # Impute missing values and compute GCS, RSBI, and VE
    chart_data = impute_values(chart_data)
    chart_data = compute_total_gcs(chart_data)
    chart_data = compute_rsbi_ve(chart_data)

    df_ser_features = compute_ser_features(chart_data,
                                           seg_size_hours=SEG_SIZE_HOURS,
                                           stats=STATS)

    # Group by label/segment to find how many data we have per label
    segment_groups = df_ser_features.groupby('label')

    # Get segments based on the total number of hours and segment size.
    num_segments = TOTAL_NUM_HOURS // SEG_SIZE_HOURS
    df_ser_feat_trun = pd.concat([df_ser_features.loc[group_idx]
                                  for i, group_idx in segment_groups.groups.items()
                                  if i <= num_segments])

    # Use only the features in the training dataset
    columns_to_keep = ['label', 'icustay_id', 'seg_start_time', 'seg_end_time']
    columns_to_keep.extend(list(ser_feat_population_mean.index.values))
    # If feature does not exist use average value.
    for column in columns_to_keep:
        if column not in df_ser_feat_trun.columns:
            df_ser_feat_trun[column] = ser_feat_population_mean[column]
    df_ser_feat_trun = df_ser_feat_trun[columns_to_keep]

    # Impute nan feature values with their averages
    segment_groups_trun = df_ser_feat_trun.groupby('label')
    feature_count_seg = segment_groups_trun.count()
    for feat in feature_count_seg.columns:
        if feat not in ('icustay_id', 'seg_start_time', 'seg_end_time'):
            # Impute nan feature values with their averages
            for label, label_idx in segment_groups_trun.groups.items():
                df_ser_feat_trun.loc[label_idx, feat] = df_ser_feat_trun.loc[label_idx, feat].fillna(
                    ser_feat_population_mean[feat])

    return df_ser_feat_trun


def predict_ser_risk(chart_data):
    # Extract trained model
    # Note that model has been trained based on eval hours, but can be extended.
    model_filename = './data/gbt_model_eval-hours{0}_seg-size-hours{1}.sav'.format(
        12, SEG_SIZE_HOURS)
    if os.path.exists(model_filename):
        model = joblib.load(model_filename)
    else:
        return None

    # Get features.
    df_ser_feat_trun = get_ser_analysis_df(chart_data)

    # If features are not None, predict labels and risk.
    if df_ser_feat_trun is not None:
        X = df_ser_feat_trun.drop(['label', 'icustay_id', 'seg_start_time',
                                'seg_end_time'], axis=1)
        y_true = df_ser_feat_trun['label']

        # Predict label (1 close to the event) and risk
        y_pred = model.predict(X)
        ser_risk = (-np.log(y_pred) + np.log(max(y_pred))) \
            / max(-np.log(y_pred) + np.log(max(y_pred)))

        df_ser_pred = pd.DataFrame({'y_pred': y_pred,
                                    'ser_risk': ser_risk,
                                    'y_true': y_true,
                                    'seg_start_time':
                                        df_ser_feat_trun['seg_start_time'],
                                    'seg_end_time':
                                        df_ser_feat_trun['seg_end_time']})

        return df_ser_pred
    else:
        return None
