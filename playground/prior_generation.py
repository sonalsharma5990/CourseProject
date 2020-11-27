import numpy as np


def get_top_topics(topic_significance, gamma_cutoff=0.95):
    """Get index of top topics."""
    return topic_significance[:, 0] > gamma_cutoff


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
    top_words = np.zeros(topic_word_prob.shape, dtype=int)
    top_words[x, y] = 1
    print(top_words.shape)
    return top_words

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


def create_word_stream(top_words, term_doc_matrix, matching_dates):
    top_words_count = np.einsum('ij,jk->ijk', top_words, term_doc_matrix)
    word_stream = top_words_count @ matching_dates
    print('word stream', word_stream.shape)
    return word_stream
