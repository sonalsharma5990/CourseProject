import numpy as np


def running_mean(timeseries, window_len):
    cumsum = np.cumsum(np.insert(timeseries, 0, 0), dtype=float)
    return (cumsum[window_len:] - cumsum[:-window_len]) / window_len


def make_stationary(timeseries, window_len=3):
    cum_average = running_mean(timeseries, window_len)
    return np.diff(cum_average)


if __name__ == '__main__':
    a = np.arange(1, 21)
    print(a)
    print(make_stationary(a))
    # print(make_stationary(a,3))
