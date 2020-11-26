import numpy as np

def get_top_words(lda_model, prob_m=0.25):
    """Get top words with cumulative probability mass cutoff."""
    topic_word_prob = lda_model.get_topics()
    # filter topics which are significant
    print(np.round(topic_word_prob[:5,],4))
    topic_word_index = np.argsort(-topic_word_prob, axis=1)
    topic_word_prob = -np.sort(-topic_word_prob, axis=1)
    print(np.round(topic_word_prob[:5,],4))
    print(topic_word_index[:5,])
    topic_word_prob_sum = np.cumsum(topic_word_prob, axis=1)
    print(topic_word_prob_sum.shape)
    print(np.count_nonzero(topic_word_prob_sum < prob_m, axis=1))



def create_word_stream(top_words_prob, corpus, doc_date):
    # for each topic
    # for each word
    # create count_stream
    # doc_mapping, word, count mapping in corpus
    # get matching doc_ids
    pass


