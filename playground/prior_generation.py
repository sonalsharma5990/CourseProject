import logging

import numpy as np
from gensim.matutils import corpus2dense
from tabulate import tabulate

from causality import calculate_significance

logger = logging.getLogger(__name__)


def get_top_topics(topic_significance, gamma_cutoff=0.95):
    """Get index of top topics."""
    return topic_significance[:, 1] > gamma_cutoff


def get_topic_lag(topic_significance, top_topics):
    """Get lag for significant topics."""
    return topic_significance[top_topics, :][
        :, 0].astype(int)


def get_top_words(lda_model, top_topics, prob_m=0.25):
    """Get top words with cumulative probability mass cutoff."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    # remember the original index
    topic_word_index = np.argsort(-topic_word_prob, axis=1)

    topic_word_prob = -np.sort(-topic_word_prob, axis=1)
    topic_word_prob_sum = np.cumsum(topic_word_prob, axis=1)

    prob_cutoff = topic_word_prob_sum <= prob_m
    x = np.nonzero(prob_cutoff)[0]
    y = topic_word_index[prob_cutoff]
    # top_words = np.zeros(topic_word_prob.shape, dtype=int)
    # top_words[x, y] = 1
    # # prob contain 570 words representation in compact form
    # # keey x, y information
    # print(np.sum(top_words), top_words.shape, np.max(y))
    return x, y

    # print(topic_word_prob_sum)

    # create adjancy_matrix with top words and top topics
    # by index of top_topics
    # index of top words for each topic

    # once we have num_topics * num_words matrix
    # create doc * num_words matrix
    # multiply to get num_topic * num_words * document_count matrix
    # multiply this with doc_id * matching_dates matrix
    # num_topics * num_words * count_for_matching_dates
    # pass it to causality score, stationary towards last axis
    # calculate everything along last axis
    # num_topics * num_words * significance, impact
    # calculate significance > 95%
    # if same topic +, - negative, threshold
    # split along axis
    # else make significance = 0
    # with new tn * word_seq calculate prior
    # feed new tn and word_seq to sequence


def filter_corpus(corpus, word_index):
    term_doc_matrix = corpus2dense(corpus, len(
        corpus.dictionary), dtype=int)
    # print(np.unique(word_index))
    return term_doc_matrix[np.unique(word_index), :]


def create_word_stream(term_doc_matrix, matching_dates):
    print('term doc matrix shape', term_doc_matrix.shape)
    word_stream = term_doc_matrix @ matching_dates
    print('word stream', word_stream.shape)
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


def get_new_topic_word_prob(new_topics, vocab_size):
    eta = np.zeros((len(new_topics), vocab_size))
    for topic_id, (index, prob) in enumerate(new_topics):
        eta[topic_id, index] = prob
    return eta


def print_topic_word_prob(new_topics, dictionary):
    for topic_id, (index, prob) in enumerate(new_topics):
        print('*' * 72)
        flat_table = []
        for i, word_id in enumerate(index):
            flat_table.append([topic_id, dictionary[word_id], prob[i]])
        print(
            tabulate(
                flat_table, [
                    'topic_id', 'word', 'prob'], tablefmt="grid"))


def process_topic_causality(
        topic_significance,
        lda_model,
        corpus,
        common_dates,
        nontext_series):
    top_significant_topics = get_top_topics(topic_significance)
    topic_lag = get_topic_lag(topic_significance, top_significant_topics)

    topic_index, word_index = get_top_words(lda_model, top_significant_topics)
    term_doc_matrix = filter_corpus(corpus, word_index)
    word_stream = create_word_stream(
        term_doc_matrix, common_dates)

    # topic_words_filter(topic_index, word_index, word_stream, topic_lag)

    word_significance = calculate_significance(
        word_stream,
        nontext_series,
        lag=np.unique(topic_lag))

    # print(word_significance)
    # print(word_significance.shape)
    new_topics = process_word_significance(word_significance,
                                           topic_lag, topic_index, word_index)
    print_topic_word_prob(new_topics, corpus.dictionary)
    return get_new_topic_word_prob(new_topics, len(corpus.dictionary))
