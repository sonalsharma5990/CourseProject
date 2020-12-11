from collections.abc import Iterable

import pandas as pd
import numpy as np


def get_impact(coef):
    """Return impact as +1, -1."""
    if coef > 0:
        return 1
    elif coef < 0:
        return -1
    return coef


def crosscorr(datax, datay, lag=0):
    """Lag-N cross correlation."""
    return datax.corr(datay.shift(lag))


def get_all_lags(from_data, to_data, lag):
    output = []
    for i in lag:
        coef = crosscorr(from_data, to_data, i)
        impact = get_impact(coef)
        output.append([i, coef, impact])
    # print(np.array(np.round(output,3)))
    return output


def best_lag(from_data, to_data, lag):
    all_lags = get_all_lags(from_data, to_data, range(1, lag + 1))
    return sorted(all_lags, key=lambda x: abs(x[1]), reverse=True)[0]


def get_pearson_correlation(from_data, to_data, lag):
    from_series = pd.Series(from_data)
    to_series = pd.Series(to_data)
    if isinstance(lag, Iterable):
        return np.array(get_all_lags(from_series, to_series, lag))

    return np.array(best_lag(from_series, to_series, lag))
