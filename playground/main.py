from datetime import time
import logging

import numpy as np
from gensim.models.ldamulticore import LdaMulticore, LdaModel

from lda_helper import get_corpus
from pre_process import normalize_iem_market
from utils import get_adjacency_matrix
from causality import calculate_significance

logging.getLogger('gensim.models.ldamodel').setLevel(logging.WARN)
logging.getLogger('gensim.models.ldamulticore').setLevel(logging.WARN)


def get_doc_date(filename):
    docs = []
    with open(filename) as f:
        for line in f:
            _, date_ = line.strip().split(',')
            docs.append(int(date_))
    return np.array(docs)


def load_date_doc_mapping(filename):
    date_doc_map = {}
    count = 0
    with open(filename) as f:
        for line in f:
            doc_id, date_ = line.strip().split(',')
            if date_ in date_doc_map:
                date_doc_map[date_].append(int(doc_id))
            else:
                date_doc_map[date_] = [doc_id]
            count += 1
    return date_doc_map, count


def process(data_folder, num_topics):
    data_file = f'{data_folder}/data.txt.gz'
    date_map_file = f'{data_folder}/doc_date_map.txt'
    # date_doc_map, num_docs = load_date_doc_mapping(date_map_file)
    corpus = get_corpus(data_file)
    eta = None
    # lda_model = LdaMulticore(corpus, num_topics=num_topics,
    #     id2word=corpus.dictionary,
    #     passes=10,
    #     iterations=100,
    #     # minimum_probability=0,
    #     random_state=12345,
    #     eta=eta)
    # lda_model.save(f'{data_folder}/lda_model')
    lda_model = LdaModel.load(f'{data_folder}/lda_model')

    iem_data = normalize_iem_market('200005', '200010')
    # print(iem_data)
    doc_date_map = get_doc_date(date_map_file)
    # print(doc_date_map)
    num_docs = doc_date_map.shape[0]

    topics = np.zeros((num_docs, num_topics))
    for doc_id, topic_prob in enumerate(lda_model.get_document_topics(corpus)):
        for topic_id, theta in topic_prob:
            topics[doc_id][topic_id] = theta
            # if topic_id in topics:
            #     topics[topic_id].append((doc_id, theta))
            # else:
            #     topics[topic_id] = [(doc_id,theta)]

    matching_dates = get_adjacency_matrix(
        doc_date_map, iem_data['Date'].to_numpy())
    # print(matching_dates.shape)
    time_series = topics.T @ matching_dates
    # print(time_series.shape)

    calculate_significance(
        time_series,
        iem_data['LastPrice'].to_numpy(),
        lag=5)

    # print(iem_data)
    # get docu_ids by dates
    # sum topic_prob for all docs in dates

    # for i, topic_prob in enumerate(lda.get_document_topics(corpus)):
    #     print('document',i)
    #     print(topic_prob)
    #     print('********************')
    # for topic in lda.top_topics(corpus):
    #     for word in topic:
    #         print(word)
    #     print('------------------------------')
    # count = 0
    # for k, v in corpus.dictionary.items():
    #     print(k,v)


if __name__ == '__main__':
    process('experiment_1', num_topics=30)
