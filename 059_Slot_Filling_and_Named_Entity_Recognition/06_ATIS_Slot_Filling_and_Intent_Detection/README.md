# ATIS.keras
Spoken Language Understanding(SLU)/Slot Filling in Keras. 

Blog post is available here: https://chsasank.github.io/spoken-language-understanding.html

Tutorial Implements RNNs in Keras to solve the Airline Travel Information System(ATIS) dataset.

Here is an example sentence and its labels from the dataset:

  Show   | flights | from |   Boston | to |  New | York|    today
  ---   | --- | --- |   --- | --- |  --- | ---|    ---
 O | O | O |B-dept | O|B-arr|I-arr|B-date

This tutorial also illustrates word embeddings.
