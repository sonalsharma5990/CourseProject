import numpy as np


def rand_init(num_docs, num_topics, num_words):
    """This will converge."""
    doc_topic_p = np.random.rand(
        num_docs, num_topics)
    doc_topic_p /= np.sum(doc_topic_p, axis=1, keepdims=True)
    topic_word_p = np.random.rand(
        num_topics, num_words)
    topic_word_p /= np.sum(topic_word_p, axis=1, keepdims=True)
    return doc_topic_p, topic_word_p


def old_init(num_docs, num_topics, num_words):
    """This will never converge."""
    doc_topic_p = np.ones(
        (num_docs, num_topics))
    doc_topic_p /= np.sum(doc_topic_p, axis=1, keepdims=True)
    topic_word_p = np.ones(
        (num_topics, num_words))
    topic_word_p /= np.sum(topic_word_p, axis=1, keepdims=True)
    return doc_topic_p, topic_word_p


def do_steps(doc_term, doc_topics_p, topic_word_p):
    topic_prob = np.einsum('ij,jk->ijk',
                           doc_topics_p,
                           topic_word_p)
    topic_prob /= np.sum(topic_prob, axis=1, keepdims=True)
    # print(topic_prob.shape)

    doc_topic_n = np.einsum('ij,ikj->ik', doc_term, topic_prob)
    doc_topic_n /= np.sum(doc_topic_n, axis=1, keepdims=True)

    topic_word_n = np.einsum(
        'ij,ikj->kj', doc_term, topic_prob)
    # print(topic_word_p.shape)
    # print(topic_word_n.shape)
    topic_word_n /= np.sum(
        topic_word_n, axis=1, keepdims=True)
    return doc_topic_n, topic_word_n


if __name__ == '__main__':
    num_docs = 10
    num_topics = 2
    num_words = 100
    np.random.seed(7672)
    doc_term = np.random.randint(0, high=10, size=(num_docs, num_words))
    doc_topic_p, topic_word_p = rand_init(num_docs, num_topics, num_words)
    # print(doc_term)
    # print(topic_word_p)
    # print(doc_topic_p)
    for i in range(10000):
        doc_topic_n, topic_word_n = do_steps(
            doc_term, doc_topic_p, topic_word_p)
        if (np.all(np.round(doc_topic_p, 6) == np.round(doc_topic_n, 6)) and np.all(
                np.round(topic_word_p, 6)) == np.all(np.round(topic_word_n, 6))):
            print('converged at ', i)
            # print(topic_word_n)
            print(np.round(doc_topic_n, 6))
            break
        doc_topic_p, topic_word_p = doc_topic_n, topic_word_n
