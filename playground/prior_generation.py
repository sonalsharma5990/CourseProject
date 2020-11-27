import logging
from gensim.corpora import dictionary

import numpy as np
from gensim.matutils import corpus2dense
from tabulate import tabulate

from causality import calculate_significance

logger = logging.getLogger(__name__)


def get_top_topics(topic_significance, gamma_cutoff=0.95):
    """Get index of top topics."""
    filtered_index = np.nonzero(topic_significance[:, 1] > gamma_cutoff)[0]
    sorted_index = np.argsort(-topic_significance[:, 1])
    return sorted_index[np.isin(sorted_index, filtered_index)]


def get_topic_lag(topic_significance, top_topics):
    """Get lag for significant topics."""
    return topic_significance[top_topics, :][
        :, 0].astype(int)


def get_top_words(lda_model, top_topics, prob_m=0.30):
    """Get top words with cumulative probability mass cutoff."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    # remember the original index
    topic_word_index = np.argsort(-topic_word_prob, axis=1)

    topic_word_prob = -np.sort(-topic_word_prob, axis=1)
    topic_word_prob_sum = np.cumsum(topic_word_prob, axis=1)

    prob_cutoff = topic_word_prob_sum <= prob_m
    topic_index = np.nonzero(prob_cutoff)[0]
    word_index = topic_word_index[prob_cutoff]
    return topic_index, word_index


def filter_corpus(corpus, word_index):
    term_doc_matrix = corpus2dense(corpus, len(
        corpus.dictionary), dtype=int)
    return term_doc_matrix[np.unique(word_index), :]


def create_word_stream(term_doc_matrix, matching_dates):
    word_stream = term_doc_matrix @ matching_dates
    return word_stream


def is_pure_impact(positive_impact, negative_impact, delta=0.1):
    positive_count = np.count_nonzero(positive_impact)
    negative_count = np.count_nonzero(negative_impact)
    return (positive_count < delta * negative_count
            or negative_count < delta * positive_count)


def filter_signf_words(word_sig, word_index, filter):
    return word_sig[filter, :], word_index[filter]


def get_sigificant_words(word_sig, word_index, gamma_cutoff=0.95):
    """Select top words based on gamma cutoff."""
    cutoff = word_sig[:, 1] > gamma_cutoff
    return filter_signf_words(
        word_sig, word_index, cutoff)


def process_impact(word_sig, word_index, delta=0.1):
    """Determine impact and split if necessary."""
    positive_impact = word_sig[:, 2] > 0
    negative_impact = word_sig[:, 2] < 0
    positive_count = np.count_nonzero(positive_impact)
    negative_count = np.count_nonzero(negative_impact)

    split_topic = None
    if (not negative_count
            or not positive_count):
        pass
    elif positive_count < delta * negative_count:
        logger.info('Mostly - words, ignoring + words')
        word_sig, word_index = filter_signf_words(
            word_sig, word_index, negative_impact)
    elif negative_count < delta * positive_count:
        logger.info('Mostly + words, ignoring - words')
        word_sig, word_index = filter_signf_words(
            word_sig, word_index, positive_impact)
    else:
        logger.info('Mixed +, - words: topic would be split.')
        full_word_index = word_index
        full_word_sig = word_sig
        word_sig, word_index = filter_signf_words(
            word_sig, word_index, positive_impact)
        split_topic = filter_signf_words(
            full_word_sig, full_word_index, negative_impact)
    return (word_sig, word_index), split_topic


def calculate_topic_prior(top_words, gamma_cutoff=0.95):
    return (top_words - gamma_cutoff) / np.sum(top_words - gamma_cutoff)


def process_word_significance(word_sig, topic_lag, topic_index, word_index):
    unique_words = np.unique(word_index)
    unique_lags = np.unique(topic_lag)

    new_topics = []
    for i, lag in enumerate(topic_lag):
        topic_words = word_index[topic_index == i]
        series_index = np.nonzero(np.isin(unique_words, topic_words))[0]
        topic_word_sig = word_sig[series_index, lag == unique_lags, :]

        # filter words with significance > .95%
        topic_word_sig, topic_words = get_sigificant_words(
            topic_word_sig, topic_words)

        if(topic_word_sig.size == 0):
            continue

        original_topic, split_topic = process_impact(
            topic_word_sig, topic_words)
        new_topics.append(
            (original_topic[1], calculate_topic_prior(original_topic[0][:, 1])))

        if split_topic:
            new_topics.append(
                (split_topic[1], calculate_topic_prior(split_topic[0][:, 1])))

    return new_topics


def get_new_topic_word_prob(new_topics, vocab_size, num_topics):
    """Creates new eta matrix."""
    if len(new_topics) > num_topics:
        num_topics = len(new_topics)
    eta = np.zeros((num_topics, vocab_size))
    for topic_id, (index, prob) in enumerate(new_topics):
        eta[topic_id, index] = prob
    return eta


def print_lda_top_topics(
        lda_model, top_topics, dictionary, max_topics=10, max_words=3):
    """Print LDA model top significant topics."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    if topic_word_prob.shape[0] < max_topics:
        max_topics = topic_word_prob.shape[0]
    # remember the original index
    topic_word_index = np.argsort(-topic_word_prob, axis=1)

    print('*' * 72)
    flat_table = []
    headers = [f'LDA TOP {max_words} WORDS IN SIGNIFICAN TOPICS']
    for i in range(max_topics):
        top_words_index = topic_word_index[i, :3]
        top_words = ' '.join([dictionary[i]
                              for i in top_words_index])
        flat_table.append([top_words])
    print(tabulate(flat_table, headers, tablefmt="grid"))


def print_topic_word_prob(new_topics, dictionary):
    """Print each word signficance probability for each topic."""
    for topic_id, (index, prob) in enumerate(new_topics):
        print('*' * 72)
        flat_table = []
        sorted_i = np.argsort(-prob)
        sorted_prob = -np.sort(-prob)

        for i, prob_item in enumerate(sorted_prob):
            flat_table.append(
                [topic_id, dictionary[index[sorted_i[i]]], prob_item])
        print(
            tabulate(
                flat_table, [
                    'topic_id', 'word', 'prob'], tablefmt="grid"))


def print_top_topics(
        new_topics, dictionary, max_topics=10, max_words=3):
    """Print top significant topics and words."""
    print('*' * 72)
    flat_table = []
    headers = [f'TOP {max_words} WORDS IN SIGNIFICAN TOPICS']
    for (index, prob) in new_topics[:max_topics]:
        sorted_i = np.argsort(-prob)
        top_words_index = index[sorted_i[:3]]
        top_words = ' '.join([dictionary[i]
                              for i in top_words_index])
        flat_table.append([top_words])
    print(tabulate(flat_table, headers, tablefmt="grid"))


def process_topic_causality(
        topic_significance, lda_model,
        corpus, common_dates, nontext_series, num_topics):
    """Get significance and probability for topic words."""
    top_significant_topics = get_top_topics(topic_significance)

    print_lda_top_topics(
        lda_model, top_significant_topics, corpus.dictionary)

    topic_lag = get_topic_lag(topic_significance, top_significant_topics)

    topic_index, word_index = get_top_words(lda_model, top_significant_topics)
    term_doc_matrix = filter_corpus(corpus, word_index)
    word_stream = create_word_stream(
        term_doc_matrix, common_dates)

    word_significance = calculate_significance(
        word_stream,
        nontext_series,
        lag=list(np.unique(topic_lag)))

    new_topics = process_word_significance(word_significance,
                                           topic_lag, topic_index, word_index)
    print_topic_word_prob(new_topics, corpus.dictionary)
    print_top_topics(new_topics, corpus.dictionary)

    return get_new_topic_word_prob(
        new_topics, len(
            corpus.dictionary), num_topics)
