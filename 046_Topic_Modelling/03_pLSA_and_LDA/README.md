# Topic_Modelling
A simple topic modeller using PLSA and LDA in Python

To run the modeller, run file topic_modeller.py, with arguments - location of corpus of documents, number of topics, maximum number of iterations to use and the algorithm to use (plsa or lda).

wikidump_parsing.py -- parses the wikipedia corpus available at , creates vocabulary out of it, counts number of documents and creates document-term matrix. The corpus consists of files containing wikipedia page summaries.

plsa.py -- Implementation of Probabilistic Latent Semantic Analysis.

lda_gibbs.py -- Implementation of Latent Dirichlet Allocation using Collapsed Gibbs Sampling.
