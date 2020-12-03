import numpy as np
from gensim.matutils import corpus2dense

def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix


class PlsaModel:

    """
    PLSA implementation with prior strength
    """

    def __init__(self, corpus,
                 document_topic_prob=None,
                 topic_word_prob=None,
                 mu=0,
                 prior_topic_word_prob=None):
        """
        Initialize empty document list.
        """
        self.likelihoods = []
        self.term_doc_matrix = corpus2dense(corpus, len(
            corpus.dictionary), dtype=int).T
        self.document_topic_prob = document_topic_prob
        self.topic_word_prob = topic_word_prob
        self.number_of_documents = self.term_doc_matrix.shape[0]
        self.vocabulary_size = self.term_doc_matrix.shape[1]
        self.mu = mu
        self.prior_topic_word_prob = prior_topic_word_prob
        self.topic_prob = None  # P(z | d, w)



    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize!
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        if not self.document_topic_prob:
            self.document_topic_prob = np.random.rand(
                self.number_of_documents, number_of_topics)
            self.document_topic_prob = normalize(self.document_topic_prob)

        if not self.topic_word_prob:
            self.topic_word_prob = np.random.rand(
                number_of_topics, self.vocabulary_size)
            self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        if not self.document_topic_prob:
            self.document_topic_prob = np.ones(
                (self.number_of_documents, number_of_topics))
            self.document_topic_prob = normalize(self.document_topic_prob)
        
        if not self.topic_word_prob:
            self.topic_word_prob = np.ones(
                (number_of_topics, self.vocabulary_size))
            self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        print("E step:")
        # k dimension
        self.topic_prob = np.einsum(
            'ij,jk->ijk',
            self.document_topic_prob,
            self.topic_word_prob)
        self.topic_prob /= np.sum(self.topic_prob, axis=1, keepdims=True)

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """

        print("M step:")
        # update P(w | z)
        topic_word_prob_n = np.einsum(
            'ij,ikj->kj', self.term_doc_matrix, self.topic_prob)
        topic_word_prob_d = np.sum(
            topic_word_prob_n, axis=1, keepdims=True)

        if self.prior_topic_word_prob and self.mu:
            topic_word_prob_n += self.mu * self.prior_topic_word_prob
            topic_word_prob_d += self.mu

        self.topic_word_prob = topic_word_prob_n / topic_word_prob_d

        self.document_topic_prob = np.einsum(
            'ij,ikj->ik', self.term_doc_matrix, self.topic_prob)
        self.document_topic_prob /= np.sum(
            self.document_topic_prob, axis=1, keepdims=True)

    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices

        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        log_likelihood = np.sum(self.term_doc_matrix * np.log(
            np.dot(self.document_topic_prob, self.topic_word_prob)))
        self.likelihoods.append(log_likelihood)

    def converge(self, number_of_topics, 
            max_iter=50, 
            epsilon=0.001,
            random=False):
        """
        Model topics.
        """
        print("EM iteration begins...")

        # P(z | d, w)
        self.topic_prob = np.zeros(
            [self.number_of_documents, number_of_topics, self.vocabulary_size], 
            dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=random)

        # Run the EM algorithm
        current_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")
            self.expectation_step()
            self.maximization_step(number_of_topics)
            self.calculate_likelihood(number_of_topics)
            print('likelihood change',
                  self.likelihoods[-1] - current_likelihood)
            if (current_likelihood and
                    (self.likelihoods[-1] - current_likelihood) < epsilon):
                break
            current_likelihood = self.likelihoods[-1]

