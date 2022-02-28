'''
This class does topic modeling using Latent Dirichlet Allocation using Collapsed Gibbs Sampling approach as discussed in --
(1) Finding Scientific Topics by Thomas L. Griffiths and Mark Steyvers, published in Proceedings of the National Academy of Sciences, 2004.

The class requires 'alpha' and 'beta'(Dirichlet hyperparameters) and keeps a tab on -->
(1) number of assignments of topic z in document d -- doc_topic_array
(2) number of assignments of word w to topic z -- topic_word_array
(3) total number of topics assigned to document d -- doc_alltopic_array
(4) total number of words assigned to topic z -- topic_allword_array
(5) a vocabulary that stores topic assignment for doc-word pair -- topic_dict
'''

import numpy as np
import random

class lda:

	def __init__(self, n_doc, vocabulary, n_topic=100, max_iter=1000):

		self.n_doc = n_doc
		self.n_topic = n_topic
		self.max_iter = max_iter
		self.vocabulary = vocabulary
		self.n_vocabulary = len(vocabulary)
		self.beta = 0.1
		self.alpha = float(50.0 / n_topic)
		self.doc_topic_array = np.zeros(shape=(self.n_doc, self.n_topic), dtype=int)
		self.topic_word_array = np.zeros(shape=(self.n_topic, self.n_vocabulary), dtype=int)
		self.doc_alltopic_array = np.zeros(shape=(self.n_doc), dtype=int)
		self.topic_allword_array = np.zeros(shape=(self.n_topic), dtype=int)
		self.topic_dict = {}

# randomly initialize every word in every document to a particular topic z
	def initialize(self):

		for d in range(self.n_doc):
			for w in range(self.n_vocabulary):
				z = random.randint(0,self.n_topic-1)
				self.doc_topic_array[d,z] += 1
				self.topic_word_array[z,w] += 1
				self.doc_alltopic_array[d] += 1
				self.topic_allword_array[z] += 1
				self.topic_dict[(d,w)] = z

# the Gibbs approximation to posterior probability p(z | w)
	def conditional_distribution(self, d, w):

		p_z = np.zeros(shape=(self.n_topic), dtype=float)

		left = (self.topic_word_array[:,w] + self.beta) / (self.topic_allword_array + self.beta * self.n_vocabulary)
		right = (self.doc_topic_array[d,:] + self.alpha) / (self.doc_alltopic_array[d] + self.alpha * self.n_topic)

		p_z = left * right

		for p in p_z:
			if p < 0: p = 0.0

		p_z /= np.sum(p_z)

		return p_z

# function to sample a topic from multinomial distribution
	def sample_index(self, p_z):

		return np.random.multinomial(1,p_z).argmax()

# phi represents p(w | z)
	def phi(self):

		num = self.topic_word_array + self.beta
		num /= np.sum(num, axis=1)[:, np.newaxis]

		return num

# gibbs sampler where we sample a topic assuming correctness of every other topic assignment in corpus and then update all stored arrays. This process is run 'max_iter' times
	def run_gibbs_sampler(self):

		self.initialize()
		
		for it in range(self.max_iter):
			print 'iteration #'+str(it)
			for d in range(self.n_doc):
				for w in range(self.n_vocabulary):
					z = self.topic_dict[(d,w)]
					self.doc_topic_array[d,z] -= 1
					self.topic_word_array[z,w] -= 1
					self.doc_alltopic_array[d] -= 1
					self.topic_allword_array[z] -= 1

					p_z = self.conditional_distribution(d, w)
					z = self.sample_index(p_z)

					self.doc_topic_array[d,z] += 1
					self.topic_word_array[z,w] += 1
					self.doc_alltopic_array[d] += 1
					self.topic_allword_array[z] += 1
					self.topic_dict[(d,w)] = z

			print self.phi()

'''
	def likelihood(self):	

		lik = 0
		
		for z in range(self.n_topic):
			lik += log_multi_beta(topic_word_array[z,:] + self.beta)
			lik -= log_multi_beta(self.beta, self.n_vocabulary)

		for d in range(self.n_doc):
			lik += log_multi_beta(doc_topic_array[d:] + self.alpha)
			lik -= log_multi_beta(self.alpha, self.n_topic)

		return lik


	def vertical_topic(width, z, doc_len):

		m = np.zeros(shape=(width, width))
		m[:,z] = int(doc_len / width)
		return m.flatten()

	def horizontal_topic(width, z, doc_len):
		
		m = np.zeros(shape=(width, width))
		m[z,:] = int(doc_len / width)
		return m.flatten()

	def gen_word_distribution(n_topic, doc_len):

		width = n_topic / 2
		n_vocab = width ** 2
		word_dist = np.zeros(shape=(n_topic, n_vocab))
		
		for k in range(width):
			word_dist[width,:] = vertical_topic(width, k, doc_len)

		for k in range(width):
			word_dist[width+k,:] = horizontal_topic(width, k, doc_len)

		for k in range(n_vocab):
			norm_sum = sum(word_dist[:,k])
			word_dist[z,k] /= norm_sum for z in range(n_topic)

		return word_dist

	def gen_document(word_dist, n_topic, n_vocabulary, doc_len, alpha):
		
		theta = np.random.mtrand.dirichlet([alpha], n_topic)
		gen_doc = np.zeros(shape=(n_vocabulary))

		for i in range(doc_len):
			z = sample_index(theta)
			w = sample_index(word_dist[z,:])
			gen_doc[v] += 1

		return gen_doc

	def gen_documents(word_dist, n_topic, n_vocabulary, doc_len, alpha, n_docs):

		doc_word_array = np.zeros(shape=(n_docs, n_vocabulary))

		for d in range(n_docs):
			doc_word_array[d,:] = gen_document(word_dist, n_topic, n_vocabulary, doc_len, alpha)

		return doc_word_array

	width = n_topic / 2
	n_vocabulary = width ** 2
	word_dist = gen_word_distribution(n_topic, doc_len)
	doc_word_array = gen_documents(word_dist, n_topic, n_vocabulary, doc_len, alpha, n_docs)
	lda_object = lda(n_doc, n_topic, max_iter)

	for it, phi in enumerate(lda_object.run_gibbs_sampler()):
		print "iteration", it
		print "likelihood", lda_object.likelihood()
'''
