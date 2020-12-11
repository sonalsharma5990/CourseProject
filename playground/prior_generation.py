import logging
from gensim.corpora import dictionary

import numpy as np
from gensim.matutils import corpus2dense
from tabulate import tabulate

from causality import calculate_significance

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
        word_prob = [(j,prob) for j, prob in enumerate(topic_word_prob[i])]
        sorted_word_prob = sorted(word_prob, key=lambda x:x[1], reverse=True)
        for k in range(1,len(sorted_word_prob)):
            sorted_word_prob[k] = (
                sorted_word_prob[k][0], 
                sorted_word_prob[k][1] + sorted_word_prob[k-1][1])
        # now return only satisfying index
        
        for k in range(len(sorted_word_prob)):
            if sorted_word_prob[k][1] > prob_m:
                break
            cutoff_indexes.append([i, k])
        
    topic_index = np.array([i for i,_ in cutoff_indexes])
    word_index = np.array([i for _,i in cutoff_indexes])
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


def is_pure_impact(positive_impact, negative_impact, delta=0.1):
    positive_count = np.count_nonzero(positive_impact)
    negative_count = np.count_nonzero(negative_impact)
    return (positive_count < delta * negative_count
            or negative_count < delta * positive_count)


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


def process_word_significance2(word_sig, topic_index, word_index):
    pass


def process_word_significance(
        word_sig,
        topic_lag,
        topic_index,
        word_index,
        corpus):
    unique_words = np.unique(word_index)
    unique_lags = np.unique(topic_lag)

    new_topics = []
    # print('word index',word_index.shape)
    # print('topic index', topic_index.shape)
    for i, lag in enumerate(topic_lag):
        topic_words = word_index[topic_index == i]

        series_index = np.nonzero(np.isin(unique_words, topic_words))[0]
        # print(word_sig.shape)
        topic_word_sig = word_sig[series_index, :]
        
        # change topic words order to match the word significance score
        topic_words = unique_words[series_index]
        # if i==0:
        #     print(topic_words)

        # filter words with significance > .95%
        topic_word_sig, topic_words = get_sigificant_words(
            topic_word_sig, topic_words)
        
        if i==0:
            print('index after filtering', topic_words)


        if(topic_word_sig.size == 0):
            continue

        # print('*'*72,'significant topics')
        # for i in topic_words:
        #     word = corpus.dictionary[i]
        #     # if word in ('oil', 'tax'):
        #     #     print('******')
        #     print(word)
        # # exit(1)
        # print('end', '*'*72,'significant topics')

        original_topic, split_topic = process_impact(
            topic_word_sig, topic_words)
        # print('original topic', original_topic[1])
        # print('*'*72, 'words after split')
        # for i in original_topic[1]:
        #     word = corpus.dictionary[i]
        #     # if word in ('oil', 'tax'):
        #     #     print('******')
        #     print(word)
        # print('end','*'*72, 'words after split')

        # print('topic, prior', calculate_topic_prior(original_topic[0][:, 1]))
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
    # print('topic lag', topic_lag)

    topic_index, word_index = get_top_words(lda_model, top_significant_topics)
    # topic_index_seq, word_index_seq = get_top_words(lda_model, top_significant_topics)

    # print(np.array_equal(topic_index, topic_index_seq))
    # print(np.array_equal(word_index, word_index_seq))
    # exit(1)

    # print(topic_index)
    # print(word_index)
    print('words from significant topics')
    # topic_word_index = np.column_stack((topic_index, word_index))
    old_index = -1
    count = 0
    for i,j in enumerate(word_index):
        if topic_index[i] != 0:
            break
        if old_index != topic_index[i]:
            count = 0
            old_index = topic_index[i]
        if count < 200:
            print('topic', topic_index[i], j, corpus.dictionary[j])
            count += 1
        # print('-'*72)
    # exit(1)
    print('end words from significant topics')
    # exit(1)
    # till this point verified

    print('all top words selected', np.unique(word_index).shape)
    unique_words, original_index = np.unique(word_index, return_index=True)
    # print('unique index', unique_words)
    # print('original index', word_index[original_index])

    term_doc_matrix = filter_corpus(corpus, word_index)
    word_stream = create_word_stream(
        term_doc_matrix, common_dates)

    word_significance = calculate_significance(
        word_stream,
        nontext_series,
        lag=5)

    topic_zero_words = word_index[topic_index == 0]
    # for i, word in enumerate(topic_zero_words):
    #     if i >= 10:
    #         break
    #     # print(corpus.dictionary[word])

    print_topic_word(num_topics, word_significance, 
        topic_index, word_index, corpus)

    for i, sigf in enumerate(word_significance):
        if unique_words[i] in topic_zero_words:

        # for j in range(num_topics):
        #     words_in_topic = word_index[topic_index == i]
        #     exit(1)
            # get the words for word_index
        # print by topics
        # topic index for each index of word_index
        # find the index of word_inde
       
        
            # print('u index', i)
            # here i is the index of the unique words
            # original_index[i] gives me the index in the word_index => not it doesn't give value
            word = corpus.dictionary[word_index[original_index[i]]]
            assert word_index[original_index[i]] == unique_words[i]
            if sigf[1] > 0.95:
                print(unique_words[i], word,sigf )
        
    new_topics = process_word_significance(
        word_significance, topic_lag, topic_index, word_index, corpus)
    # print_topic_word_prob(new_topics, corpus.dictionary)
    
    for i in new_topics[0][0]:
        print(i, corpus.dictionary[i])
    print('-'*72)
    for i in new_topics[1][0]:
        print(i, corpus.dictionary[i])

    # print_top_topics(new_topics, corpus.dictionary, max_words=10)
    exit(1)
    return get_new_topic_word_prob(
        new_topics, len(
            corpus.dictionary), num_topics)


def print_topic_word(num_topics, word_significance, topic_index, word_index, corpus):
    unique_words = np.unique(word_index)
    for i in range(1):
        topic_words = word_index[topic_index == i]
        print('unfiltered', topic_words)
        # if not np.count_nonzero(topic_words):
        #     break
        # for j, word in enumerate(topic_words):
        #     if j < 10:
        #         print(corpus.dictionary[word])
        print('-'*72)
        if i == 0:
            sigf_words = np.nonzero(np.isin(unique_words, topic_words))[0]
            assert np.array_equal(np.sort(unique_words[sigf_words]),np.sort(topic_words))
            # print('sigf', sigf_words)
            topic_word_sig = word_significance[sigf_words, :]
            print('topic', i)
            for k in range(topic_word_sig.shape[0]):
                if topic_word_sig[k,:][1] > 0.95:
                    print(unique_words[sigf_words[k]], corpus.dictionary[unique_words[sigf_words[k]]], topic_word_sig[k,:])
            print('-'*72)