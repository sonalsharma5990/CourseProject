import logging

import numpy as np
from gensim.matutils import corpus2dense


from causality import calculate_significance
from print_utils import (
    print_lda_top_topics,
    print_word_significance,
    print_topic_word_prob,
    print_top_topics)


logger = logging.getLogger(__name__)


def get_top_topics(topic_significance, gamma_cutoff=0.95):
    """Get index of top topics."""
    # print(topic_significance.shape)
    filtered_index = np.nonzero(topic_significance[:, 1] > gamma_cutoff)[0]
    # sorting as we want most causal topic on top
    sorted_index = np.argsort(-topic_significance[:, 1])
    return sorted_index[np.isin(sorted_index, filtered_index)]


def get_topic_lag(topic_significance, top_topics):
    """Get lag for significant topics."""
    return topic_significance[top_topics, :][
        :, 0].astype(int)


def get_top_words_seq(lda_model, top_topics, prob_m=0.40):
    """Get top words with cumulative prob cutoff using loop."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    num_topics = topic_word_prob.shape[0]
    cutoff_indexes = []
    # for each topic create a word_list with index and prob
    for i in range(len(num_topics)):
        word_prob = [(j, prob) for j, prob in enumerate(topic_word_prob[i])]
        sorted_word_prob = sorted(word_prob, key=lambda x: x[1], reverse=True)
        for k in range(1, len(sorted_word_prob)):
            sorted_word_prob[k] = (
                sorted_word_prob[k][0],
                sorted_word_prob[k][1] + sorted_word_prob[k - 1][1])
        # now return only satisfying index

        for k in range(len(sorted_word_prob)):
            if sorted_word_prob[k][1] > prob_m:
                break
            cutoff_indexes.append([i, k])

    topic_index = np.array([i for i, _ in cutoff_indexes])
    word_index = np.array([i for _, i in cutoff_indexes])
    return topic_index, word_index


def get_top_words(lda_model, top_topics, prob_m=0.40):
    """Get top words with cumulative probability mass cutoff."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    # remember the original index
    topic_word_index = np.argsort(-topic_word_prob, axis=1)

    topic_word_prob = -np.sort(-topic_word_prob, axis=1)
    topic_word_prob_sum = np.cumsum(topic_word_prob, axis=1)

    prob_cutoff = topic_word_prob_sum <= prob_m
    topic_index = np.nonzero(prob_cutoff)[0]
    # return the original word_index
    word_index = topic_word_index[prob_cutoff]
    return topic_index, word_index


def filter_corpus(corpus, word_index):
    term_doc_matrix = corpus2dense(corpus, len(
        corpus.dictionary), dtype=int)
    return term_doc_matrix[np.unique(word_index), :]


def create_word_stream(term_doc_matrix, matching_dates):
    # print(matching_dates)
    word_stream = term_doc_matrix @ matching_dates
    # print(word_stream[:10,:2])
    return word_stream


def filter_signf_words(word_sig, word_index, filter_):
    return word_sig[filter_, :], word_index[filter_]


def get_sigificant_words(word_sig, word_index, gamma_cutoff=0.95):
    """Select top words based on gamma cutoff."""
    # print('word signficance shape', word_sig[:3,:])
    cutoff = word_sig[:, 1] > gamma_cutoff
    # print('cutoff', np.nonzero(cutoff)[0])
    # print('word index', word_index.shape)

    # filtered_index = np.nonzero(word_sig[:, 1] > gamma_cutoff)[0]
    # # sorting as we want most causal topic on top
    # sorted_index = np.argsort(-word_sig[:, 1])
    # filter_ = sorted_index[np.isin(sorted_index, filtered_index)]
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


def get_word_significance_by_topic(
        topic_id,
        word_sig,
        topic_index,
        word_index):
    """Get word significance by topic."""
    unique_words = np.unique(word_index)
    topic_words = word_index[topic_index == topic_id]

    # get sorted unique index which is used to index word significance
    series_index = np.nonzero(np.isin(unique_words, topic_words))[0]
    # get word signifance for this topic
    topic_word_sig = word_sig[series_index, :]

    # change topic words order to match the word significance score
    topic_words = unique_words[series_index]

    # filter words with significance > .95%
    topic_word_sig, topic_words = get_sigificant_words(
        topic_word_sig, topic_words)
    return topic_word_sig, topic_words


def get_topic_word_significance(
        top_significant_topics,
        word_sigf,
        topic_index,
        word_index):
    """Get word significance for top significant topics."""
    old_topics = []
    for i in range(top_significant_topics.shape[0]):
        topic_word_sig, topic_words = get_word_significance_by_topic(
            i, word_sigf, topic_index, word_index)
        if(topic_word_sig.size == 0):
            continue
        old_topics.append((topic_word_sig, topic_words))
    return old_topics


def split_topics_impact(old_topics):
    """Split word significance if required."""
    new_topics = []
    for topic_word_sig, topic_words in old_topics:
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
    # print('Print final output')
    for topic_id, (index, prob) in enumerate(new_topics):
        # print(topic_id, index, prob)
        eta[topic_id, index] = prob
    return eta


def process_topic_causality(
        topic_significance, lda_model,
        corpus, common_dates, nontext_series, num_topics):
    """Get significance and probability for topic words."""
    top_significant_topics = get_top_topics(topic_significance)

    print_lda_top_topics(
        lda_model, top_significant_topics, corpus.dictionary)

    # topic_lag = get_topic_lag(topic_significance, top_significant_topics)
    # print('topic lag', topic_lag)

    topic_index, word_index = get_top_words(lda_model, top_significant_topics)
    # topic_index_seq, word_index_seq = get_top_words(lda_model, top_significant_topics)

    term_doc_matrix = filter_corpus(corpus, word_index)
    word_stream = create_word_stream(
        term_doc_matrix, common_dates)

    word_sigf = calculate_significance(
        word_stream,
        nontext_series,
        lag=5)

    old_topics = get_topic_word_significance(
        top_significant_topics, word_sigf, topic_index, word_index)

    new_topics = split_topics_impact(old_topics)

    # print split
    print_word_significance(old_topics, corpus.dictionary)
    print_topic_word_prob(new_topics, corpus.dictionary)

    # print significant topics with significant words
    print_top_topics(new_topics, corpus.dictionary, max_words=10)

    return get_new_topic_word_prob(
        new_topics, len(
            corpus.dictionary), num_topics)
