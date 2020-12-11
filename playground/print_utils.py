from tabulate import tabulate
import numpy as np


def print_lda_top_topics(
        lda_model, top_topics, dictionary, max_topics=10, max_words=3):
    """Print LDA model top significant topics."""
    topic_word_prob = lda_model.get_topics()[top_topics, :]

    if topic_word_prob.shape[0] < max_topics:
        max_topics = topic_word_prob.shape[0]
    # remember the original index
    topic_word_index = np.argsort(-topic_word_prob, axis=1)

    print('')
    flat_table = []
    headers = [f'LDA TOP {max_words} WORDS IN SIGNIFICAN TOPICS']
    for i in range(max_topics):
        top_words_index = topic_word_index[i, :3]
        top_words = ' '.join([dictionary[i]
                              for i in top_words_index])
        flat_table.append([top_words])
    print(tabulate(flat_table, headers, tablefmt="grid"))


def get_largest_word_len(words_index, dictionary):
    """Return the length of largest word."""
    # print(words_index)
    return max([len(dictionary[i])
                for i in words_index])


def print_topic_word_prob(new_topics, dictionary):
    """Print each word signficance probability for each topic."""
    words_index = [i
                   for topic_words, _ in new_topics
                   for i in topic_words]
    filler_word = '-' * get_largest_word_len(words_index, dictionary)
    headers = ['WORD', 'PROB']
    flat_table = []
    print('')
    for index, prob in new_topics:
        sorted_i = np.argsort(-prob)
        sorted_prob = -np.sort(-prob)
        sorted_prob = np.round(sorted_prob, 2)
        # sorted_prob = np.round(prob, 2)
        for i, prob_item in enumerate(sorted_prob):
            flat_table.append(
                [dictionary[index[sorted_i[i]]], prob_item])
        flat_table.append([filler_word, '------'])
    print(tabulate(flat_table, headers, tablefmt='github'))


def print_top_topics(
        new_topics, dictionary, max_topics=10, max_words=3):
    """Print top significant topics and words."""
    print('')
    flat_table = []
    headers = [f'TOP {max_words} WORDS IN SIGNIFICAN TOPICS']
    for (index, prob) in new_topics[:max_topics]:
        sorted_i = np.argsort(-prob)
        top_words_index = index[sorted_i[:3]]
        top_words = ' '.join([dictionary[i]
                              for i in top_words_index])
        flat_table.append([top_words])
    print(tabulate(flat_table, headers, tablefmt="grid"))


def sort_word_signficance(
        word_sigf, topic_words, filter, impact, dictionary):
    """Sort words by significance."""
    word_sigf = word_sigf[filter, :]
    table = []
    sorted_index = np.argsort(-word_sigf[:, 1])
    sorted_index = range(len(word_sigf))
    for i in sorted_index:
        word_index = topic_words[i]
        sig_per = int(word_sigf[i][1] * 100)
        table.append([dictionary[word_index], impact, sig_per])
    return table


def print_word_significance(old_topics, dictionary):
    """Print word significance by topic."""
    words_index = [i for _, topic_words in old_topics
                   for i in topic_words]
    filler_word = '-' * get_largest_word_len(words_index, dictionary)
    headers = ['WORD', 'IMPACT', 'SIG (%)']
    table = []
    for topic_word_sig, topic_words in old_topics:
        # print(topic_word_sig.shape)
        # sorted_index = np.argsort(-topic_word_sig[:, 2])
        for i, word_index in enumerate(topic_words):
            # word_index = topic_words[i]
            impact = '+'
            if topic_word_sig[i][2] < 0:
                impact = '-'
            word_sig = int(topic_word_sig[i][1] * 100)
            table.append([dictionary[word_index], impact, word_sig])

        # for i in range(topic_word_sig):
        # if topic_word_sig[:, 2] > 0:
        #     impact = '+'
        # else:
        #     impact = '-'
        # positive_impact = topic_word_sig[:, 2] > 0
        # negative_impact = topic_word_sig[:, 2] < 0
        # table.extend(sort_word_signficance(
        #     topic_word_sig, topic_words, 
        #     positive_impact, '+', dictionary))
        # table.extend(sort_word_signficance(
        #     topic_word_sig, topic_words, 
        #     negative_impact, '-', dictionary))
        # for i in topic_words:

        table.append([filler_word, '--------', '---------'])
    print(tabulate(table, headers, tablefmt='github'))


def print_topic_word(word_significance, topic_index, word_index, dictionary):
    """Debug method to compare topic words signifance."""
    unique_words = np.unique(word_index)
    for i in range(1):
        topic_words = word_index[topic_index == i]
        print('unfiltered', topic_words)
        # if not np.count_nonzero(topic_words):
        #     break
        # for j, word in enumerate(topic_words):
        #     if j < 10:
        #         print(corpus.dictionary[word])
        print('-' * 72)
        if i == 0:
            sigf_words = np.nonzero(np.isin(unique_words, topic_words))[0]
            assert np.array_equal(
                np.sort(
                    unique_words[sigf_words]),
                np.sort(topic_words))
            # print('sigf', sigf_words)
            topic_word_sig = word_significance[sigf_words, :]
            print('topic', i)
            for k in range(topic_word_sig.shape[0]):
                if topic_word_sig[k, :][1] > 0.95:
                    print(unique_words[sigf_words[k]],
                          dictionary[unique_words[sigf_words[k]]],
                          topic_word_sig[k, :])
            print('-' * 72)


def print_significant_words(topic_index, word_index, dictionary):
    """Debug method for significant topic words."""
    print('words from significant topics')
    # topic_word_index = np.column_stack((topic_index, word_index))
    old_index = -1
    count = 0
    for i, j in enumerate(word_index):
        if topic_index[i] != 0:
            break
        if old_index != topic_index[i]:
            count = 0
            old_index = topic_index[i]
        if count < 200:
            print('topic', topic_index[i], j, dictionary[j])
            count += 1
        # print('-'*72)
    # exit(1)
    print('end words from significant topics')


def print_first_topic_words(
        num_topics,
        topic_index,
        word_index,
        corpus,
        word_significance):
    """Print signficant words for first topic."""
    print('all top words selected', np.unique(word_index).shape)
    unique_words, original_index = np.unique(word_index, return_index=True)
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
            # original_index[i] gives me the index in the word_index => not it
            # doesn't give value
            word = corpus.dictionary[word_index[original_index[i]]]
            assert word_index[original_index[i]] == unique_words[i]
            if sigf[1] > 0.95:
                print(unique_words[i], word, sigf)
