"""This module calculates the quality measures for topic sign."""
import numpy as np


def calculate_average_significance(topics):
    """Calculate average significance of all topics."""
    len_sigf_words = 0
    total_sigf = 0
    for topic_word_sig, _ in topics:
        len_sigf_words += topic_word_sig.shape[0]
        total_sigf += np.sum(100 * topic_word_sig[:, 1])
    return total_sigf / len_sigf_words


def calculate_topic_purity(topic_word_sig):
    """Calculate purity of topic."""
    num_words = topic_word_sig.shape[0]
    p_prob = np.count_nonzero(topic_word_sig[:, 2] > 0) / num_words
    n_prob = np.count_nonzero(topic_word_sig[:, 2] < 0) / num_words
    entropy_topic = p_prob * np.log(p_prob) + n_prob * np.log(n_prob)
    purity = 100 + 100 * entropy_topic
    # print(purity)
    return purity


def calculate_average_purity(topics):
    """Calculate average purity of all topics."""
    len_purity = len(topics)
    total_purity = 0
    for topic_word_sigf, _ in topics:
        total_purity += calculate_topic_purity(topic_word_sigf)
    return total_purity / len_purity
