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


if __name__ == '__main__':
    a = np.arange(1, 21)
    b = np.arange(10, 21)
    get_adjacency_matrix(a, b)
