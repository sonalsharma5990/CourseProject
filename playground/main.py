import logging
import sys

import numpy as np
from gensim.models.ldamulticore import LdaMulticore, LdaModel


from lda_helper import (
    get_corpus,
    get_document_topic_prob,
    print_lda_topics)
from pre_process import (
    normalize_iem_market,
    match_dates,
    save_topic_stats)
from utils import get_adjacency_matrix
from causality import calculate_topic_significance
from prior_generation import process_topic_causality
from timeseries import create_theta_timeseries

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s',
    level=logging.INFO)
logging.getLogger('gensim.models.ldamodel').setLevel(logging.WARN)
logging.getLogger('gensim.models.ldamulticore').setLevel(logging.WARN)
logging.getLogger('gensim.utils').setLevel(logging.WARN)


def get_doc_date(filename):
    """Get document dates array."""
    docs = []
    with open(filename) as f:
        for line in f:
            _, date_ = line.strip().split(',')
            docs.append(int(date_))
    return np.array(docs)


def get_iem_data(doc_date_map):
    iem_data = normalize_iem_market('200005', '200010')
    doc_dates = list(set(doc_date_map))
    # missing dates 20000607 20000608 in IEM data
    return match_dates(iem_data, doc_dates)


def get_nontext_series(data_folder):
    """Initialize timeseries for experiment 1."""
    date_map_file = f'{data_folder}/doc_date_map.txt'
    doc_date_map = get_doc_date(date_map_file)
    iem_data = get_iem_data(doc_date_map)
    # doc date matrix i=doc, j=date i,j=1 if doc exists for date
    doc_date_matrix = get_adjacency_matrix(
        doc_date_map, iem_data['Date'].to_numpy())
    nontext_series = iem_data['LastPrice'].to_numpy()
    return doc_date_matrix, nontext_series


def load_corpus(data_folder):
    """Initialize corpus and timeseries for experiment 1."""
    data_file = f'{data_folder}/data.txt.gz'
    return get_corpus(data_file)


def train_lda_model(
        corpus, num_topics, iter_i,
        eta=None, mu=0, load_saved=False):
    if load_saved:
        lda_model = LdaModel.load(f'experiment_1/lda_model_{iter_i}')
    else:
        lda_model = LdaMulticore(corpus, num_topics=num_topics,
                                 id2word=corpus.dictionary,
                                 passes=10,
                                 iterations=100,
                                 decay=mu,
                                 # minimum_probability=0,
                                 # random_state=98765432,
                                 eta=eta)
        # lda_model.save(f'experiment_1/lda_model_{iter_i}')
    logger.info('LDA model built.')
    return lda_model


def process_exp1(lda_model, corpus, doc_date_matrix, nontext_series,
                 num_docs, num_topics):
    """Process experiment-1."""
    doc_topic_prob = get_document_topic_prob(
        lda_model, corpus, num_docs, num_topics)

    topics_signf = calculate_topic_significance(
        doc_topic_prob, doc_date_matrix, nontext_series)

    # print(np.array_equal(np.round(topic_timeseries, 6), np.round(topics_signf, 6)))

    # print(topics_signf)

    return process_topic_causality(
        topics_signf,
        lda_model,
        corpus,
        doc_date_matrix,
        nontext_series,
        num_topics)


def experiment_1(exp_mu=50, num_topics=30, load_saved=False):
    data_folder = 'experiment_1'
    corpus = load_corpus(data_folder)
    doc_date_matrix, nontext_series = get_nontext_series(data_folder)
    num_docs = sum(1 for _ in corpus)
    topic_stats = []

    # initial mu and eta
    eta = None
    mu = 0
    lda_model = None
    for i in range(5):
        logger.info('Processing iteration %s with t_n %s and mu %s',
                    i + 1, num_topics, mu)
        print('Iteration', i + 1)
        lda_model = train_lda_model(
            corpus, num_topics, i,
            eta=eta, mu=mu, load_saved=load_saved)
        eta, avg_sigf, avg_purity = process_exp1(
            lda_model, corpus, doc_date_matrix, nontext_series,
            num_docs, num_topics)
        # print(np.sum(eta, axis=1))
        mu = exp_mu
        topic_stats.append([
            exp_mu, num_topics, i + 1,
            avg_sigf, avg_purity])
    print_lda_topics(lda_model, num_topics=10, max_words=3)
    return topic_stats


def experiment_1_eval():
    all_mu = [10, 50, 100, 500, 1000]
    mu_topic_stats = []
    for mu in all_mu:
        mu_topic_stats.extend(experiment_1(exp_mu=mu))

    save_topic_stats('experiment_1/mu_stats.csv', mu_topic_stats)

    tn_topic_stats = []
    all_tn = [10, 20, 30, 40]
    for tn in all_tn:
        tn_topic_stats.extend(experiment_1(num_topics=tn))
    save_topic_stats('experiment_1/tn_stats.csv', tn_topic_stats)


if __name__ == '__main__':
    command = ''
    if sys.argv[1:]:
        command = sys.argv[1].strip()
        if command == 'retrain':
            experiment_1(load_saved=False)
        elif command == 'graph':
            experiment_1_eval()
        else:
            print('Invalid options', sys.argv[1:])
            print('Usage:')
            print(
                'python main.py         '
                '# load trained model and run timeseries feedback.')
            print(
                'python main.py retrain '
                '# retrain model and run timeseries feedback.')
            print(
                'python main.py graph   '
                '# retrain model and run timeseries feedback to produce graphs.')
            exit(1)
    experiment_1(load_saved=True)
