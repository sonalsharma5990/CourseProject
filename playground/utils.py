"""This module contains utility functions."""
import numpy as np


def get_indices(a, b):
    """Get indices of matching elements of b in a."""
    sorter = np.argsort(b)
    return sorter[np.searchsorted(b, a, sorter=sorter)]


def get_adjacency_matrix(a, b):
    """Creates an adjancy matrix for common elements in a and b."""
    adj_matrix = np.zeros((a.shape[0], b.shape[0]), dtype=int)
    common_elements = np.isin(a, b)
    a_index = np.nonzero(common_elements)[0]
    b_index = get_indices(a[common_elements], b)
    adj_matrix[a_index, b_index] = 1
    return adj_matrix


def running_mean(a, window_len, axis=None):
    """Calculate running mean along axis."""
    cumsum = np.cumsum(np.insert(a, 0, 0, axis=axis),
                       axis=axis,
                       dtype=float)
    size = np.size(a, axis=axis)
    return (np.take(
        cumsum, range(window_len, size + 1),
        axis=axis) - np.take(
            cumsum, range(0, size - window_len + 1),
            axis=axis)) / window_len


def make_stationary(a, window_len, axis=None):
    """
    Make array `a` stationary by rolling average and subtracting elements.
    """
    cum_average = running_mean(a, window_len, axis=axis)
    if axis is None:
        return np.diff(cum_average)
    return np.diff(cum_average, axis=axis)


if __name__ == '__main__':
    a = np.arange(1, 21)
    b = np.arange(10, 21)
    print('a', a)
    # get_adjacency_matrix(a, b)
    # print(a.reshape(-1))
    print(make_stationary(a, 3, axis=0))
    print(make_stationary(a, 3))
    # print(make_stationary(np.arange(1, 11), 3))
    # print(make_stationary(np.arange(11, 21), 3))
