pubmed-docsim
=============

pubmed-docsim provides tools for document similarity analysis over a corpus of medical documents. Training a model for topic modeling is very resource and time intensive, once such a model is created however using it to do similarity analysis over other medical documents is far easier. Therefore, the goal is to allow for models to be created using the freely available and very exhaustive Pubmed Open Access (OA) dataset. Pre-trained models can then be used to carry out document similarity on a corpus of other medical documents.  


Tools provided here
-------------------

- train-pubmed-lsi-models.py 

This allows you to create a model using Latent Semantic Indexing (LSI) over the Pubmed Open Access (OA) dataset. The Pubmed documents should be in the NXML format. It is also possible to do similarity analysis over these Pubmed documents as you train the model.


- lsi-docsim-using-pubmed-models.py 

This allows you to use a pretrained model created using the above tool to do document similarity over a corpus of other medical documents. PDF and NXML format documents are supported.


Pre-trained Model
-----------------

A pre-trained LSI model created over the Pubmed Open Access (OA) dataset licensed for commercial use is provided at the following location :

https://www.dropbox.com/s/imsymndys6149y0/pretrained-pubmed-lsi-models.tar.gz?dl=0

You can choose to use the above model instead of training your own. Parameters used within train-pubmed-lsi-models.py to create this model are as follows :

num_topics = 500 	# number of topics.
min_words = 256         # ignore documents shorter than 256 characters. 

The above tar.gz should be extracted to a convenient location, the directory can be pointed to while running lsi-docsim-using-pubmed-models.py.


Usage Example
-------------

Usage instructions for both the above tools can be accessed using the -h command line directive.


For Training:

python ../train-pubmed-lsi-models.py ~/Documents/Corpora/PubMedOAXMLData/ --sortsims perdoc >> results.txt


For topic modeling on your own corpus :

python3 lsi-docsim-using-pubmed-models.py ./pretrained-models/ pdf ~/Documents/biorxiv-research-papers/ >> results-20092018.txt


License
-------

The code has been licensed under GPLv3. You are free to use the pre-trained model provided by the link above in any non-commercial or commercial use as you may see fit.



External Resources
------------------

- Pubmed Open Access Dataset Download links :

https://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/
ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk

- Pubmed Parser

git+https://github.com/titipata/pubmed_parser.git








