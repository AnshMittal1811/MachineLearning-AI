'''
This file is the main file that takes following command line inputs -

(1) input_dir_path --> directory path for your text files of the corpus
(2) max_iter --> number of iterations until which you want Expectation-Maximisation in probabilistic lsa to run
(3) n_topics --> number of topics into which the corpus needs to be clustered
(4) algorithm --> mention which algorithm to be used for topic modeling - 'plsa' or 'lda'

This file creates an object of WikiDump class, provides it with input_dir_path and uses it to create vocabulary over the corpus, count number of documents and create the term-document matrix. Then based on which algorithm you want to use, it creates an object of the respective class and does the topic modeling over  the corpus.
'''

import sys
from wikidump_parsing import WikiDump
from plsa import prob_lsa
from lda_gibbs import lda

arg_list = sys.argv
input_dir_path = arg_list[1]															# input directory path of corpus
max_iter = int(arg_list[2])																# maximum number of iterations for EM algorithm
n_topics = int(arg_list[3])																# number of topics into which corpus is to be clustered
algorithm = arg_list[4]																	# either 'plsa' or 'lda'

wiki_obj = WikiDump(input_dir_path, 2)
vocabulary = wiki_obj.gen_vocabulary()													# vocabulary over provided corpus
n_docs = wiki_obj.count_docs()															# number of documents in corpus
n_vocabulary = len(vocabulary)															# size of vocabulary
term_doc_array = wiki_obj.create_term_doc_matrix(n_docs, n_vocabulary)					# term-document matrix over corpus

if algorithm is 'plsa':
	prob_lsa_obj = prob_lsa(vocabulary, n_docs, n_vocabulary, term_doc_array)				
	prob_lsa_obj.EM_algo(max_iter, n_topics)											# running plsa over provided corpus

else:
	lda_obj = lda(n_docs, vocabulary, n_topics, max_iter)
	print 'starting gibbs sampler'
	lda_obj.run_gibbs_sampler()
