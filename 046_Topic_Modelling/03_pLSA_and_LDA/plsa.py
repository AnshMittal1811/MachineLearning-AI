'''
This class implements the Probabilistic Latent Semantic Analysis algorithm as described in -->
Unsupervised Learning by Probabilistic Latent Semantic Analysis by Thomas Hofmann, published in the Journal of Machine Learning, 2001
'''

import random
import numpy as np

class prob_lsa:	

	def __init__(self, vocabulary, n_docs, n_vocabulary, term_doc_array):

		print 'Inside plsa'
		self.vocabulary = vocabulary
		self.n_docs = n_docs
		self.n_vocabulary = n_vocabulary
		self.term_doc_array = term_doc_array

		#-------------------  EM Algorithm  -------------------#

	def EM_algo(self, max_iter, n_topic):		
	
		print 'Inside EM'
		topic_term_prob = np.zeros(shape=(n_topic, self.n_vocabulary), dtype = float)
		doc_topic_prob = np.zeros(shape=(self.n_docs, n_topic), dtype = float)
		topic_prob = np.zeros(shape=(self.n_docs, self.n_vocabulary, n_topic), dtype = float)

		for z in range(n_topic):
			for w in range(self.n_vocabulary):
				topic_term_prob[z,w] = random.random()

		for d in range(self.n_docs):
			for z in range(n_topic):
				doc_topic_prob[d,z] = random.random()

		print 'initial topic_term_prob -->'
		print topic_term_prob

		print 'initial doc_topic_prob -->'
		print doc_topic_prob

		for iteration in range(max_iter):
			print 'iteration #'+str(iteration)

			#------  E-step  ---------#
	
			for d in range(self.n_docs):														# iterating over every document	
				for w in range(self.n_vocabulary):												# iterating over all words in the document
					prod_total = doc_topic_prob[d,:] * topic_term_prob[:,w]						# normalizing factor								
					#if prod_total == 0: continue
					for z in range(n_topic):													# calculate prob for all topics word can belong to
						topic_prob[d,w,:] = (doc_topic_prob[d,z] * topic_term_prob[z,w])


			#------  M-step  ---------#

			for z in range(n_topic):
				prod_sum = sum([(self.term_doc_array[:,w] * topic_prob[:,w,z]) for w in range(self.n_vocabulary)])
				#if prod_sum == 0: continue	   		
				for w in range(self.n_vocabulary):
					s = 0
					for d in range(self.n_docs):				
			   			s += (self.term_doc_array[d,w] * topic_prob[d,w,z])
					topic_term_prob[z][w] = s   		 
			   		 	
			for d in range(self.n_docs):
		   		doc_length = sum(self.term_doc_array[d,:])
		   	 	for z in range(n_topic):
					s = 0
					for w in range(self.n_vocabulary):
		   				s += (self.term_doc_array[d,w] * topic_prob[d,w,z])
					doc_topic_prob[d][z] = s

		print 'final topic_prob -->'
		print topic_prob	

		print 'final topic_term_prob -->'
		print topic_term_prob

		print 'final doc_topic_prob -->'
		print doc_topic_prob
