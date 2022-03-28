# import library
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import pathlib
import plotly
import plotly.express as px
import simdkalman

from utils.appLogger import log


def calc_haversine(lat1, lon1, lat2, lon2):
    """Calculates the great circle distance between two points
    on the earth. Inputs are array-like and specified in decimal degrees.
    """
    RADIUS = 6_367_000
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    dist = 2 * RADIUS * np.arcsin(a**0.5)
    return dist


def add_distance_diff(df):
    df['latDeg_prev'] = df['latDeg'].shift(1)
    df['latDeg_next'] = df['latDeg'].shift(-1)
    df['lngDeg_prev'] = df['lngDeg'].shift(1)
    df['lngDeg_next'] = df['lngDeg'].shift(-1)
    df['phone_prev'] = df['phone'].shift(1)
    df['phone_next'] = df['phone'].shift(-1)

    df['dist_prev'] = calc_haversine(
        df['latDeg'], df['lngDeg'], df['latDeg_prev'], df['lngDeg_prev'])
    df['dist_next'] = calc_haversine(
        df['latDeg'], df['lngDeg'], df['latDeg_next'], df['lngDeg_next'])

    df.loc[df['phone'] != df['phone_prev'], [
        'latDeg_prev', 'lngDeg_prev', 'dist_prev']] = np.nan
    df.loc[df['phone'] != df['phone_next'], [
        'latDeg_next', 'lngDeg_next', 'dist_next']] = np.nan

    return df


def get_train_score(df, gt):
    gt = gt.rename(columns={'latDeg': 'latDeg_gt', 'lngDeg': 'lngDeg_gt'})
    df = df.merge(gt, on=['collectionName', 'phoneName',
                  'millisSinceGpsEpoch'], how='inner')
    # calc_distance_error
    df['err'] = calc_haversine(
        df['latDeg_gt'], df['lngDeg_gt'], df['latDeg'], df['lngDeg'])
    # calc_evaluate_score
    df['phone'] = df['collectionName'] + '_' + df['phoneName']
    res = df.groupby('phone')['err'].agg([percentile50, percentile95])
    res['p50_p90_mean'] = (res['percentile50'] + res['percentile95']) / 2
    score = res['p50_p90_mean'].mean()
    return score


def percentile50(x):
    return np.percentile(x, 50)


def percentile95(x):
    return np.percentile(x, 95)


def make_shifted_matrix(vec):
    matrix = []
    size = len(vec)
    for i in range(size):
        row = [0] * i + vec[:size - i]
        matrix.append(row)
    return np.array(matrix)


def make_state_vector(T, size):
    vector = [1, 0]
    step = 2
    for i in range(size - 2):
        if i % 2 == 0:
            vector.append(T)
            T *= T / step
            step += 1
        else:
            vector.append(0)
    return vector


def make_noise_vector(noise, size):
    noise_vector = []
    for i in range(size):
        if i > 0 and i % 2 == 0:
            noise *= 0.5
        noise_vector.append(noise)
    return noise_vector


def make_kalman_filter(T, size, noise, obs_noise):
    vec = make_state_vector(T, size)
    state_transition = make_shifted_matrix(vec)
    process_noise = np.diag(make_noise_vector(
        noise, size)) + np.ones(size) * 1e-9
    observation_model = np.array(
        [[1] + [0] * (size - 1), [0, 1] + [0] * (size - 2)])
    observation_noise = np.diag([obs_noise] * 2) + np.ones(2) * 1e-9
    kf = simdkalman.KalmanFilter(
        state_transition=state_transition,
        process_noise=process_noise,
        observation_model=observation_model,
        observation_noise=observation_noise)
    return kf


best_params = [[1.5, 2, 2.1714133956113952e-06, 1.719317114286542e-05]]
#best_params = [[0.9352900312712755, 2, 7.050183540617244e-06, 2.917496229668982e-05]]
for i in range(0, len(best_params)):
    T, half_size, noise, obs_noise = best_params[i]


kf = make_kalman_filter(T, half_size * 2, noise, obs_noise)


def apply_kf_smoothing(df, kf_=kf):
    unique_paths = df[['collectionName', 'phoneName']
                      ].drop_duplicates().to_numpy()
    for collection, phone in unique_paths:
        cond = np.logical_and(df['collectionName'] ==
                              collection, df['phoneName'] == phone)
        data = df[cond][['latDeg', 'lngDeg']].to_numpy()
        data = data.reshape(1, len(data), 2)
        smoothed = kf_.smooth(data)
        df.loc[cond, 'latDeg'] = smoothed.states.mean[0, :, 0]
        df.loc[cond, 'lngDeg'] = smoothed.states.mean[0, :, 1]
    return df


def make_lerp_data(df):
    '''
    Generate interpolated lat,lng values for different phone times in the same collection.
    '''
    org_columns = df.columns

    # Generate a combination of time x collection x phone and combine it with the original data (generate records to be interpolated)
    time_list = df[['collectionName', 'millisSinceGpsEpoch']].drop_duplicates()
    phone_list = df[['collectionName', 'phoneName']].drop_duplicates()
    tmp = time_list.merge(phone_list, on='collectionName', how='outer')

    lerp_df = tmp.merge(
        df, on=['collectionName', 'millisSinceGpsEpoch', 'phoneName'], how='left')
    lerp_df['phone'] = lerp_df['collectionName'] + '_' + lerp_df['phoneName']
    lerp_df = lerp_df.sort_values(['phone', 'millisSinceGpsEpoch'])

    # linear interpolation
    lerp_df['latDeg_prev'] = lerp_df['latDeg'].shift(1)
    lerp_df['latDeg_next'] = lerp_df['latDeg'].shift(-1)
    lerp_df['lngDeg_prev'] = lerp_df['lngDeg'].shift(1)
    lerp_df['lngDeg_next'] = lerp_df['lngDeg'].shift(-1)
    lerp_df['phone_prev'] = lerp_df['phone'].shift(1)
    lerp_df['phone_next'] = lerp_df['phone'].shift(-1)
    lerp_df['time_prev'] = lerp_df['millisSinceGpsEpoch'].shift(1)
    lerp_df['time_next'] = lerp_df['millisSinceGpsEpoch'].shift(-1)
    # Leave only records to be interpolated
    lerp_df = lerp_df[(lerp_df['latDeg'].isnull()) & (lerp_df['phone'] == lerp_df['phone_prev']) & (
        lerp_df['phone'] == lerp_df['phone_next'])].copy()
    # calc lerp
    lerp_df['latDeg'] = lerp_df['latDeg_prev'] + (0.95 * (lerp_df['latDeg_next'] - lerp_df['latDeg_prev']) * (
        (lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))
    lerp_df['lngDeg'] = lerp_df['lngDeg_prev'] + (0.95 * (lerp_df['lngDeg_next'] - lerp_df['lngDeg_prev']) * (
        (lerp_df['millisSinceGpsEpoch'] - lerp_df['time_prev']) / (lerp_df['time_next'] - lerp_df['time_prev'])))

    # Leave only the data that has a complete set of previous and next data.
    lerp_df = lerp_df[~lerp_df['latDeg'].isnull()]

    return lerp_df[org_columns]


def calc_mean_pred(df, lerp_df):
    '''
    Make a prediction based on the average of the predictions of phones in the same collection.
    '''
    add_lerp = pd.concat([df, lerp_df])
    mean_pred_result = add_lerp.groupby(['collectionName', 'millisSinceGpsEpoch'])[
        ['latDeg', 'lngDeg']].mean().reset_index()
    mean_pred_df = df[['collectionName',
                       'phoneName', 'millisSinceGpsEpoch']].copy()
    mean_pred_df = mean_pred_df.merge(mean_pred_result[['collectionName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg']], on=[
                                      'collectionName', 'millisSinceGpsEpoch'], how='left')
    return mean_pred_df


def get_removedevice(input_df: pd.DataFrame, divece: str) -> pd.DataFrame:
    input_df['index'] = input_df.index
    input_df = input_df.sort_values('millisSinceGpsEpoch')
    input_df.index = input_df['millisSinceGpsEpoch'].values

    output_df = pd.DataFrame()
    for _, subdf in input_df.groupby('collectionName'):

        phones = subdf['phoneName'].unique()

        if (len(phones) == 1) or (not divece in phones):
            output_df = pd.concat([output_df, subdf])
            continue

        origin_df = subdf.copy()

        _index = subdf['phoneName'] == divece
        subdf.loc[_index, 'latDeg'] = np.nan
        subdf.loc[_index, 'lngDeg'] = np.nan
        subdf = subdf.interpolate(
            method='index', limit_area='inside', limit_direction='both')

        _index = subdf['latDeg'].isnull()
        subdf.loc[_index, 'latDeg'] = origin_df.loc[_index, 'latDeg'].values
        subdf.loc[_index, 'lngDeg'] = origin_df.loc[_index, 'lngDeg'].values

        output_df = pd.concat([output_df, subdf])

    output_df.index = output_df['index'].values
    output_df = output_df.sort_index()

    del output_df['index']

    return output_df


def processCoordinates(collectionName, phoneName, msSinceEpoch, latitude, longitude, phone, trial=True):
    th = 50
    bt = [collectionName, phoneName, msSinceEpoch, latitude, longitude, phone]
    col = ['collectionName', 'phoneName',
           'millisSinceGpsEpoch', 'latDeg', 'lngDeg', 'phone']
    # if len(base_test) == 1:
    if trial == True:
        base_test = pd.DataFrame(columns=[
                                 'collectionName', 'phoneName', 'millisSinceGpsEpoch', 'latDeg', 'lngDeg', 'phone'])
        base_test.loc[0, :] = bt
        base_test.to_csv(f'{collectionName}_df.csv', index=False)

        return latitude, longitude

    else:
        base_test = pd.read_csv(f'{collectionName}_df.csv')
        print(base_test.shape)
        # base_test.loc[len(base_test)-1,:] = bt
        # base_test.append(bt)
        base_test = pd.concat([base_test, pd.DataFrame(
            [bt], columns=col)], axis=0).reset_index()
        base_test = add_distance_diff(base_test)

        base_test.loc[((base_test['dist_prev'] > th) & (
            base_test['dist_next'] > th)), ['latDeg', 'lngDeg']] = np.nan

        test_kf = apply_kf_smoothing(base_test)

        test_lerp = make_lerp_data(test_kf)
        test_mean_pred = calc_mean_pred(test_kf, test_lerp)

        phones = list(test_mean_pred['phoneName'].unique())
        test_remove = get_removedevice(test_mean_pred, phones[0])
        for phone in phones[1:]:
            test_remove = get_removedevice(test_remove, phone)

        test_remove['NewlatDeg'] = test_remove['latDeg']
        test_remove['NewlngDeg'] = test_remove['lngDeg']

        test_remove.to_csv(f'{collectionName}_df.csv', index=False)

        log(len(test_remove))

        return test_remove.loc[-1, 'latDeg'], test_remove.loc[-1, 'lngdeg']


# for i in range(0,15):
#     tryy = False
#     print(i)
#     # if i == 1:
#         # bt = ['gurgaon','pixel',i,37.416646 , -122.082040, 'pixel']

#     bt = ['gurgaon','pixel',i,37.416628 , -122.082053, 'pixel']
#     if i ==0:
#         tryy = True

#     pp_boii(bt = bt,th = 1, trial= tryy)
