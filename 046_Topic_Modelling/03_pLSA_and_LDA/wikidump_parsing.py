'''
The Wikidump class is used to create 'vocabulary' of words from the wikipedia corpus provided by 'input_dir_path', count number of documents in the corpus and create a term-document matrix from the corpus. I have removed non-alphabetic tokens, stopwords mentioned in nltk English corpus andd words that have frequency less than 'min_freq' in corpus. 
'''

import sys
import os
import re
import glob
from collections import OrderedDict
import numpy as np
from nltk import RegexpTokenizer
from nltk.corpus import stopwords

class WikiDump:

	def __init__(self, input_dir_path, min_freq=5):
		self.input_dir_path = input_dir_path
		self.min_freq = min_freq
		self.local_vocabulary = OrderedDict({})
		self.tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
		self.stopword_set = set(stopwords.words('english'))

# This function generates a vocabulary of words with thier frequency in the provided corpus.

	def clean_vocab(self):

		for key, value in self.local_vocabulary.iteritems():
			if value < self.min_freq or key in self.stopword_set: del self.local_vocabulary[key]

	def gen_vocabulary(self):	
		
		for filename in glob.glob(os.path.join(self.input_dir_path, '*.txt')):
			with open(filename, 'r') as fi: 											
				lines = fi.readlines()
				for line in lines:				
					if line[0] == '<' or line == '': continue
					else: 
						line = line.lower()
						word_list = self.tokenizer.tokenize(line)
						for word in word_list: 
							if word in self.local_vocabulary: self.local_vocabulary[word] += 1
							else: self.local_vocabulary[word] = 1
		
		self.clean_vocab()
		print 'vocabulary --> '+str(self.local_vocabulary)
		return self.local_vocabulary

# This function counts the number of documents in the provided corpus.

	def count_docs(self):
		n_doc = 0	
		for filename in glob.glob(os.path.join(self.input_dir_path, '*.txt')):
			with open(filename, 'r') as fi:
				lines = fi.readlines()
				for line in lines:
					if line[:4] == '<doc': n_doc += 1
					else: continue
	
		return n_doc 

# This function creates a term-document matrix from the provided corpus.

	def create_term_doc_matrix(self, n_doc, n_vocabulary):
		print 'creating term-doc matrix...'		
		term_doc_array = np.zeros(shape=(n_doc, n_vocabulary), dtype=int)
		d = 0 		
		for filename in glob.glob(os.path.join(self.input_dir_path, '*.txt')):
			with open(filename, 'r') as fi:
				lines = fi.readlines()
				for line in lines:								
					if line[:3] == '<doc': 					
						continue
					elif line[:] == '</doc>':
						d += 1
						continue
					else:
						line = line.lower()
						word_list = self.tokenizer.tokenize(line)						
						for word in word_list:
							if word in self.local_vocabulary: 
								w = self.local_vocabulary.keys().index(word)
								term_doc_array[d,w] += 1
							else: continue
		
		return term_doc_array					

