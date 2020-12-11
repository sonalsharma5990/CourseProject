"""This module contains method for timeseries manipulation."""
import numpy as np


def create_theta_timeseries(doc_topic_prob, doc_dates_map):
    """Creates topic timeseries using doc-topic prob theta."""
    topic_timeseries  = []
    num_topics = doc_topic_prob.shape[1]
    
    # for each topic a new timeseries
    for j in range(num_topics):
        theta_timeseries = []
        old_date_ = doc_dates_map[0]
        theta_sum = 0
        for i, date_ in enumerate(doc_dates_map):
            if old_date_ != date_:
                theta_timeseries.append(theta_sum)
                theta_sum = 0
                old_date_ = date_
            theta_sum += doc_topic_prob[i][j]
        theta_timeseries.append(theta_sum)
        topic_timeseries.append(theta_timeseries)
    print(np.array(topic_timeseries).shape)
    print(np.array(topic_timeseries))
    return np.array(topic_timeseries)

def create_word_timeseries():
    """Create word timeseries using word counts per document."""
    pass