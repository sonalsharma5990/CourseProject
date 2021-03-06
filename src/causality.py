from collections.abc import Iterable

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

from utils import make_stationary
from pearson_correlation import get_pearson_correlation


def get_impact(gc_result, best_lag):
    impact = np.sum(
        gc_result[best_lag][1][1].params[best_lag:-1]) / np.abs(best_lag)
    if impact > 0:
        return 1
    elif impact < 0:
        return -1
    return impact


def best_lag(gc_result):
    min_p_value = float('inf')
    best_lag = 0
    for k, v in gc_result.items():
        p_value = v[0]['ssr_ftest'][1]
        if p_value < min_p_value:
            best_lag = k
            min_p_value = 1 - p_value
    significance = min_p_value
    impact = get_impact(gc_result, best_lag)
    return np.array([best_lag, significance, impact])


def all_lags(gc_result):
    output = []
    for lag, v in gc_result.items():
        p_value = v[0]['ssr_ftest'][1]
        significance = 1 - p_value
        impact = get_impact(gc_result, lag)
        output.append([lag, significance, impact])
    return np.array(output)


def get_significance(from_data, to_data, lag):
    """Run granger test and accumulate results."""
    data = np.stack((to_data, from_data), axis=1)
    gc_result = grangercausalitytests(data, lag, verbose=False)
    if isinstance(lag, Iterable):
        return all_lags(gc_result)
    return best_lag(gc_result)


def normalize_causality(causality):
    """Normalize significance for all enteries."""
    # for each lag
    print(causality.shape)
    if len(causality.shape) == 3:
        print(causality[:10])
        print(-np.sort(-causality[:, :, 1], axis=0)[:10])
        most_significant = -np.sort(-causality[:, :, 1], axis=0)[0]
        print('3d most sign', most_significant)
        causality[:, :, 1] /= most_significant

    else:
        print(-np.sort(-causality[:, 1])[:10])
        most_significant = -np.sort(-causality[:, 1])[0]
        causality[:, 1] /= most_significant

    print(np.round(causality[0], 5))

    return causality


def calculate_significance(
        from_timeseries, to_timeseries, lag, method='granger'):
    """Calculate significance between timeseries.

    Parameters
    ----------
    method : {'granger', 'pearson'}
    lag : int, default 5
    """
    if method == 'granger':
        # granger test needs stationary series
        from_timeseries = make_stationary(
            from_timeseries, window_len=3, axis=1)
        to_timeseries = make_stationary(to_timeseries, window_len=3)
        func = get_significance
    else:
        func = get_pearson_correlation

    return np.apply_along_axis(
        func,
        1,
        from_timeseries,
        to_timeseries,
        lag)
    # return normalize_causality(causality)


def calculate_topic_significance(
        topics, doc_date_matrix, nontext_series, lag=5, method='granger'):
    """Calculate topic significance based on lag and method."""
    time_series = topics.T @ doc_date_matrix
    # print(time_series.shape)
    # return time_series
    return calculate_significance(
        time_series,
        nontext_series,
        lag=lag,
        method=method)


def calculate_topic_significance_seq(doc_topic_prob, doc_date):
    """Calculate topic significance using for loop."""

    # get dates from experiment1 data
    # get dates form IEM data
    # compare if any date missing
    # if not map with doc id in the document
    # and multiply
    # for every topic, for each day, sum all documents theta_j
    # for every date
    # find all doc_index
    # sum the topic_distribution

    pass
