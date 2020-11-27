import logging

import numpy as np
from gensim.models.ldamulticore import LdaMulticore, LdaModel


from lda_helper import get_corpus, get_document_topic_prob
from pre_process import normalize_iem_market
from utils import get_adjacency_matrix
from causality import calculate_topic_significance
from prior_generation import process_topic_causality


logging.getLogger('gensim.models.ldamodel').setLevel(logging.WARN)
logging.getLogger('gensim.models.ldamulticore').setLevel(logging.WARN)
logging.getLogger('gensim.utils').setLevel(logging.WARN)


def get_doc_date(filename):
    docs = []
    with open(filename) as f:
        for line in f:
            _, date_ = line.strip().split(',')
            docs.append(int(date_))
    return np.array(docs)


def get_nontext_series(data_folder):
    """Initialize timeseries for experiment 1."""
    date_map_file = f'{data_folder}/doc_date_map.txt'
    iem_data = normalize_iem_market('200005', '200010')
    doc_date_map = get_doc_date(date_map_file)

    common_dates = get_adjacency_matrix(
        doc_date_map, iem_data['Date'].to_numpy())
    nontext_series = iem_data['LastPrice'].to_numpy()
    return common_dates, nontext_series


def initialize_exp1(data_folder):
    """Initialize corpus and timeseries for experiment 1."""
    data_file = f'{data_folder}/data.txt.gz'
    corpus = get_corpus(data_file)
    return corpus, get_nontext_series(data_folder)


def process_exp1(corpus, common_dates, nontext_series, num_topics,
                 eta=None):
    """Process experiment-1."""
    # lda_model = LdaMulticore(corpus, num_topics=num_topics,
    #     # id2word=corpus.dictionary,
    #     passes=10,
    #     iterations=100,
    #     # minimum_probability=0,
    #     random_state=12345,
    #     eta=eta)
    lda_model = LdaModel.load(f'experiment_1/lda_model')
    topics = get_document_topic_prob(
        lda_model, corpus, num_topics)
    topics_signf = calculate_topic_significance(
        topics, common_dates, nontext_series)
    return process_topic_causality(
        topics_signf,
        lda_model,
        corpus,
        common_dates,
        nontext_series)


def experiment_1():
    corpus, (common_dates, nontext_series) = initialize_exp1(
        'experiment_1')
    eta = None
    mu = 50
    for i in range(1):
        eta = process_exp1(
            corpus, common_dates, nontext_series,
            num_topics=30, eta=eta)
        if eta is not None:
            eta = mu * eta


if __name__ == '__main__':
    experiment_1()
