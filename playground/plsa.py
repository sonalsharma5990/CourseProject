import logging

import numpy as np
from numpy.core.numeric import count_nonzero
import sparse
from gensim.matutils import corpus2csc


class PlsaModel:

    """
    PLSA implementation with prior strength
    """

    def __init__(self,
                 corpus,
                 num_topics,
                 doc_topic_prob=None,
                 topic_word_prob=None,
                 mu=0,
                 prior_topic_word_prob=None):
        """
        Initialize empty document list.
        """
        self.likelihoods = []
        self.num_topics = num_topics

        csc = corpus2csc(corpus, dtype=int)
        # replace nan to 0
        print(np.any(np.isnan(csc.todense())))
        csc.data = np.nan_to_num(csc.data)
        self.term_doc_matrix = sparse.COO.from_scipy_sparse(csc).T

        self.doc_topic_prob = doc_topic_prob
        self.topic_word_prob = topic_word_prob
        print(len(corpus.dictionary))
        print(self.term_doc_matrix.shape)
        self.num_docs = self.term_doc_matrix.shape[0]
        self.num_terms = self.term_doc_matrix.shape[1]
        self.mu = mu
        self.prior_topic_word_prob = prior_topic_word_prob
        self.topic_prob = None  # P(z | d, w)

    def initialize_randomly(self, min_prob):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob
        """
        min_prob = 1e-5
        if not self.doc_topic_prob:
            doc_topic_prob = np.random.rand(
                self.num_docs, self.num_terms)
            doc_topic_prob /= np.sum(
                doc_topic_prob, axis=1, keepdims=True)
            print(doc_topic_prob[1])
            print(np.count_nonzero(doc_topic_prob < min_prob))

            doc_topic_prob[doc_topic_prob < min_prob] = 0

            self.doc_topic_prob = sparse.COO(doc_topic_prob)

        if not self.topic_word_prob:
            topic_word_prob = np.random.rand(
                self.num_topics, self.num_terms)
            topic_word_prob /= np.sum(
                topic_word_prob, axis=1, keepdims=True)
            topic_word_prob[topic_word_prob < min_prob] = 0
            self.topic_word_prob = sparse.COO(topic_word_prob)

    # handle differently for first iteration

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        # k dimension
        # can be replaced as a.reshape(*a.shape,1) * b
        self.topic_prob = np.einsum(
            'ij,jk->ijk',
            self.document_topic_prob,
            self.topic_word_prob)
        self.topic_prob /= np.sum(self.topic_prob, axis=1, keepdims=True)

    def em_step(self):
        """EM step."""
        print('shape of doc topic prob', self.doc_topic_prob.shape)
        print('shape of topic_word_prob', self.topic_word_prob.shape)
        topic_prob = self.doc_topic_prob.reshape(
            (*self.doc_topic_prob.shape, 1)) * self.topic_word_prob.T
        topic_prob /= topic_prob.sum(axis=1, keepdims=True)

        # print(self.term_doc_matrix.todense())
        self.doc_topic_prob = sparse.diagonal(
            topic_prob.dot(self.term_doc_matrix),
            axis2=2).T
        self.doc_topic_prob /= np.sum(
            self.doc_topic_prob, axis=1, keepdims=True)

        topic_word_prob_n = sparse.diagonal(
            np.dot(topic_prob.T, self.term_doc_matrix), axis2=2)

        topic_word_prob_d = np.sum(
            topic_word_prob_n, axis=1, keepdims=True)

        # Handle mu and prior strength
        if self.prior_topic_word_prob and self.mu:
            topic_word_prob_n += self.mu * self.prior_topic_word_prob
            topic_word_prob_d += self.mu

        self.topic_word_prob = topic_word_prob_n / topic_word_prob_d

    # handle differently for first iteration
    # handle differently if one-dim matrix

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """

        print("M step:")
        # update P(w | z)
        # np.diagonal((c.T @ d),axis2=2)
        # can be replaced by 6*np.sum(d,axis=0) for first uniform run
        topic_word_prob_n = np.einsum(
            'ij,ikj->kj', self.term_doc_matrix, self.topic_prob)
        topic_word_prob_d = np.sum(
            topic_word_prob_n, axis=1, keepdims=True)

        if self.prior_topic_word_prob and self.mu:
            topic_word_prob_n += self.mu * self.prior_topic_word_prob
            topic_word_prob_d += self.mu

        self.topic_word_prob = topic_word_prob_n / topic_word_prob_d

        # can be replaced by 6*np.sum(d,axis=1,keepdims=True) for first uniform run
        # np.diagonal((c @ d.T),axis2=2).T
        self.document_topic_prob = np.einsum(
            'ij,ikj->ik', self.term_doc_matrix, self.topic_prob)
        self.document_topic_prob /= np.sum(
            self.document_topic_prob, axis=1, keepdims=True)

    def calculate_likelihood(self):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        Append the calculated log-likelihood to self.likelihoods

        """
        return np.sum(self.term_doc_matrix * np.log(
            np.dot(self.document_topic_prob, self.topic_word_prob)))

    def iterate(self, min_prob, max_iter, epsilon):
        # Run the EM algorithm
        min_prob = max(min_prob, 1e-8)
        self.initialize_randomly(min_prob)
        current_likelihood = 0.0
        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.em_step()
            new_likelihood = self.calculate_likelihood()
            print('likelihood change',
                  new_likelihood - current_likelihood)
            if (current_likelihood and
                    (new_likelihood - current_likelihood) < epsilon):
                return new_likelihood
            current_likelihood = new_likelihood
        return current_likelihood

    def converge(self,
                 min_prob=0.01,
                 passes=10,
                 max_iter=50,
                 epsilon=0.001,
                 random=False):
        """
        Model topics.
        """
        # self.build_term_doc_matrix()
        if self.doc_topic_prob and self.topic_word_prob:
            # if probabilities are already provided, no passes
            passes = 1

        for i in range(passes):
            likelihood = self.iterate(min_prob, max_iter, epsilon)
            self.likelihoods.append(
                [likelihood, self.doc_topic_prob, self.topic_word_prob])
            self.doc_topic_prob = None
            self.topic_word_prob = None
