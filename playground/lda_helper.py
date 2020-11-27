"""This module contains helper functions for LDA modeling."""
import gzip

import numpy as np
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    remove_stopwords,
    strip_punctuation,
    preprocess_string)


def get_tokens(path):
    """Create tokens after applying filters."""
    filters = [lambda x: x.lower(),
               strip_punctuation,
               remove_stopwords]
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


def get_corpus(path):
    """Create dictionary and lazy load corpus."""
    dictionary = Dictionary(get_tokens(path))
    return NYTimesCorpus(path, dictionary)


def get_document_topic_prob(lda_model, corpus, num_docs, num_topics):
    topics = np.zeros((num_docs, num_topics))
    for doc_id, topic_prob in enumerate(lda_model.get_document_topics(corpus)):
        for topic_id, theta in topic_prob:
            topics[doc_id][topic_id] = theta
    return topics
