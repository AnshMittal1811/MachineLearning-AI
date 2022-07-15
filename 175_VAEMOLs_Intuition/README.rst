Variational Autoencoder for Molecules
***********************************************

Variational autoencoder for molecules in tensorflow.

Dependencies
============

1. Rdkit

.. code:: shell

    conda install -c rdkit rdkit

2. Tensorflow

cpu-version

.. code:: shell

    pip install tensorflow

gpu-version

.. code:: shell

    pip install tensorflow-gpu


Preprocessing
=============

1. Data
-------

`ChEBML 24 Database <https://www.ebi.ac.uk/chembl/downloads>`_
was used for SMILES data.

SMILES strings were padded with spaces to max_len(default=120) and strings larger than max_len were discarded. Remaining strings are labeled character by character(max_len labels in one string).

2. preprocess.py
----------------

Does the following steps:

1. Downloads `chembl_24_1_chemreps.txt.gz <ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_24_1_chemreps.txt.gz>`_

2. Preprocess SMILES strings

3. Saves processed data into numpy arrays.

Numpy arrays contains training data, testing data, dictionaries for character <-> label(integer) interchange.

Training
========

1. Model
--------

Model consists of CNN encoder and CuDNNGRU decoder and defined in 
`vae.py <https://github.com/YunjaeChoi/vaemols/blob/master/vaemols/models/vae.py>`_

2. train.py
-----------

Does the following steps:

1. Loads preprcessed data

2. trains with fit_generator using DataGenerator


Notebooks
=========

Notebooks are here to help after training is done.

1. `structure_variation.ipynb <https://github.com/YunjaeChoi/vaemols/blob/master/structure_variation.ipynb>`_
-------------------------------------------------------------------------------------------------------------

This notebook helps to get variational structures when given a SMILES string.

2. `visualize_latent_space.ipynb <https://github.com/YunjaeChoi/vaemols/blob/master/visualize_latent_space.ipynb>`_
-------------------------------------------------------------------------------------------------------------------

This notebook helps visualizing learned latent space using a plot or tensorboard.

tensorboard visualization example:

.. image:: https://raw.githubusercontent.com/YunjaeChoi/vaemols/master/doc/image/tensorboard.png

3. `find_top_k_mols_in_latent_space.ipynb <https://github.com/YunjaeChoi/vaemols/blob/master/find_top_k_mols_in_latent_space.ipynb>`_
-------------------------------------------------------------------------------------------------------------------------------------

This notebook helps to get top_k similar molecules measured by euclidean distance in latent space.



