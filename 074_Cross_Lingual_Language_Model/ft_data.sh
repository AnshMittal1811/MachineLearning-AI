#!/bin/bash

# Usage : ft_data.sh $languages

set -e

languages=$1

# File containing the codes and vocabularies from the previous meta-processing. 
CODE_VOCAB_PATH=

# path where processed files will be stored
OUTPATH=/content/processed

# If parallel data is available and you need to preprocess it
PARA=True
# If you want to process monolingual data (if the monolingual data is unavailable and you 
# leave this parameter set to True, the parallel data will be used to build the monolingual data)
MONO=True    
# folder containing the parallel data
PARA_PATH=/content/data/para
# folder containing the monolingual data
MONO_PATH=/content/data/para

# Percentage of data to use as test data (%)
test_size=10 
# Percentage of data to use as validation data (%)
val_size=10   

# tools paths
TOOLS_PATH=tools
TOKENIZE=$TOOLS_PATH/tokenizer_our.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

# create output path
mkdir -p $OUTPATH
# avoid permission error
chmod +x $FASTBPE
chmod +x $TOOLS_PATH/mosesdecoder/scripts/tokenizer/*.perl

# The n_sample parameter is optional, and when it is not passed or when it exceeds the dataset size, the whole dataset is considered
n_samples=-1

# transform (tokenize, lower and remove accent, loard code and vocab, apply BPE tokenization, binarize...) our data contained 
# in the text files into a pth file understandable by the framework.

# Let's consider the sub-task en-fr.
chmod +x build_fine_tune_data.sh
. ../scripts/build_fine_tune_data.sh $languages $n_samples
