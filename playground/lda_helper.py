"""This module contains helper functions for LDA modeling."""
import gzip

from tabulate import tabulate
import numpy as np

from gensim import utils
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    STOPWORDS,
    remove_stopwords,
    strip_punctuation,
    strip_numeric,
    strip_short,
    preprocess_string)


EXP1_STOPWORDS = STOPWORDS.union(set(['gore', 'bush', 'said']))


def exp1_remove_stopwords(s):
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w not in EXP1_STOPWORDS)


def get_tokens(path, exp='exp1'):
    """Create tokens after applying filters."""
    if exp == 'exp1'
    stopwords_func = exp1_remove_stopwords
    else:
        stopwords_func = remove_stopwords
    filters = [lambda x: x.lower(),
               strip_punctuation,
               strip_numeric,
               remove_stopwords,
               strip_short]
    for line in gzip.open(path, 'rt'):
        yield preprocess_string(line, filters=filters)


class NYTimesCorpus:
    """Lazy loaded NYTimes corpus."""

    def __init__(self, path, dictionary):
        self.path = path
        self.dictionary = dictionary

    def __iter__(self):
        for tokens in get_tokens(self.path):
            yield self.dictionary.doc2bow(tokens)


def get_corpus(path, exp='exp1'):
    """Create dictionary and lazy load corpus."""
    dictionary = Dictionary(get_tokens(path, exp=exp))
    return NYTimesCorpus(path, dictionary)


def get_document_topic_prob(lda_model, corpus, num_docs, num_topics):
    topics = np.zeros((num_docs, num_topics))
    for doc_id, topic_prob in enumerate(lda_model.get_document_topics(corpus)):
        for topic_id, theta in topic_prob:
            topics[doc_id][topic_id] = theta
    return topics


def print_lda_topics(lda_model, num_topics, max_words=10):
    topics = lda_model.show_topics(num_topics=num_topics,
                                   num_words=max_words, formatted=False)
    print('*' * 72)
    flat_table = []
    headers = [f'LDA TOP {max_words} WORDS IN TOPICS']
    for topic_id, words_idx in topics:
        words = ' '.join(i
                         for i, _ in words_idx)
        flat_table.append([words])
    print(tabulate(flat_table, headers, tablefmt="grid"))
