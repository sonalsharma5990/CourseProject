import logging

from gensim.models.ldamulticore import LdaMulticore

from lda_helper import get_corpus
from pre_process import normalize_iem_market

logging.getLogger('gensim.models.ldamodel').setLevel(logging.WARN)
logging.getLogger('gensim.models.ldamulticore').setLevel(logging.WARN)

def load_date_doc_mapping(filename):
    date_doc_map = {}
    with open(filename) as f:
        for line in f:
            doc_id, date_ = line.strip().split(',')
            if date_ in date_doc_map:
                date_doc_map[date_].append(int(doc_id))
            else:
                date_doc_map[date_] = [doc_id]
    return date_doc_map




def process(data_folder, num_topics):
    data_file = f'{data_folder}/data.txt.gz'
    date_map_file = f'{data_folder}/doc_date_map.txt'
    date_doc_map = load_date_doc_mapping(date_map_file)
    corpus = get_corpus(data_file)
    eta = None
    lda_model = LdaMulticore(corpus, num_topics=num_topics,
        id2word=corpus.dictionary,
        # iterations=500,
        eta=eta)

    iem_data = normalize_iem_market()

    
    
    
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
    process('experiment_1',num_topics=30)