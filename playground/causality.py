import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from utils import make_stationary


def get_impact(gc_result, best_lag):
    impact = np.sum(
        gc_result[best_lag][1][1].params[best_lag:-1]) / np.abs(best_lag)
    if impact > 0:
        return 1
    elif impact < 0:
        return -1
    raise ValueError('No impact found.')


def get_significance(from_data, to_data, lag):
    # print('single column from shape', from_data.shape)
    # print('single column to shape', to_data.shape)
    data = np.stack((to_data, from_data), axis=1)
    # print('output shape', data.shape)
    # print(data)
    gc_result = grangercausalitytests(data, lag, verbose=False)
    min_p_value = float('inf')
    best_lag = 0
    for k, v in gc_result.items():
        p_value = v[0]['ssr_ftest'][1]
        if p_value < min_p_value:
            best_lag = k
            min_p_value = p_value
    significance = 1 - min_p_value
    impact = get_impact(gc_result, best_lag)
    return np.array([significance, impact])


def calculate_significance(from_timeseries, to_timeseries, lag):
    from_timeseries = make_stationary(from_timeseries, window_len=3, axis=1)
    to_timeseries = make_stationary(to_timeseries, window_len=3)
    # print('from_timeseries shape:', from_timeseries.shape)
    # print('to_timeseries shape:', to_timeseries.shape)
    return np.apply_along_axis(
        get_significance,
        1,
        from_timeseries,
        to_timeseries,
        lag)
