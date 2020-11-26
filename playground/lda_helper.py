import gzip

from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import (
    remove_stopwords,
    strip_punctuation,
    preprocess_string)
from gensim.models.ldamulticore import LdaMulticore


def get_tokens(path, func=None):
    filters = [lambda x: x.lower(),
               strip_punctuation,
               remove_stopwords]
    for line in gzip.open(path, 'rt'):
        if func:
            yield func(preprocess_string(
                line, filters=filters))
        else:
            yield preprocess_string(line, filters=filters)


class NYTimesCorpus:
    def __init__(self, path, dictionary):
        self.path = path
        self.dictionary = dictionary

    def __iter__(self):
        for tokens in get_tokens(self.path):
            yield self.dictionary.doc2bow(tokens)        


def get_corpus(path):
    dictionary = Dictionary(get_tokens(path))
    return NYTimesCorpus(path, dictionary)


