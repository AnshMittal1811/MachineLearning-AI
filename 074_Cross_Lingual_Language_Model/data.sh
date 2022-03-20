#!/bin/bash

# usage : data.sh $languages
# Transform (tokenize, lower and remove accent, loard code and vocab, learn and apply BPE tokenization,
# binarize...) our data contained in the text files into a pth file understandable by the framework : 
# takes a lot of time with dataset size, nCodes and shuf_n_samples

set -e

# languages 
lgs=$1
 
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
# whether all languages should share the same vocabulary (leave to True)
SAME_VOCAB=True    
# The following parameter allows, when having independent monolingual and parallel data, to add the parallel data to the monolingual data. It is left by default to True. 
add_para_data_to_mono_data=False 

# Learn nCodes BPE code on the training data
nCodes=20000
# Generating shuf_n_samples random permutations of training data to learn bpe
shuf_n_samples=10000 
# It is preferable and advisable that it be the powers of two...
threads_for_tokenizer=16 
# Percentage of data to use as test data (%)
test_size=10 
# Percentage of data to use as validation data (%)
val_size=10              

# tools paths
TOOLS_PATH=tools
TOKENIZE=$TOOLS_PATH/tokenizer_our.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast
#PROCESSED_FILE=../scripts/build_meta_data_multixlm.sh

# The n_sample parameter is optional, and when it is not passed or when it exceeds the dataset size, the whole dataset is considered
n_samples=-1
# If you don't have any other data to fine-tune your model on a specific sub-task, specify the percentage of the sub-task metadata to consider or -1 to ignore it.
#sub_tasks=en-fr:10,de-en:-1,de-fr:-1
#If you want the subtasks to be constructed from the pair combinations of your languages, put the three dots
sub_tasks=...
tasks_n_samples=-1

##############################################

function abrev() {
    # todo
    result=$1
 }

if [ $sub_tasks="..." ]; then
    sub_tasks=""
	IFS=', ' read -r -a langs_array <<< "$languages"
	# todo : sort the array in alphebical oder
	array_length=${#langs_array[*]}
	for (( i=0; i<$array_length; ++i)); do 
		for (( j=$(($i+1)); j<$array_length; ++j)); do
            abrev ${langs_array[$i]} 
            a=$result
            abrev ${langs_array[$j]} 
            b=$result
        	sub_tasks=$sub_tasks,$a-$b:$tasks_n_samples
		done
	done
	# Remove the comma in front
	sub_tasks=$(echo $sub_tasks | cut -c2-)
fi

echo $sub_tasks

# create output path
mkdir -p $OUTPATH
# avoid permission error
chmod +x $FASTBPE
chmod +x $TOOLS_PATH/mosesdecoder/scripts/tokenizer/*.perl

echo "======================="
echo "Processed"
echo "======================="

### option 1 : data in the different folders with the name $pair for each pair ###
#chmod +x ../scripts/build_meta_data_multixlm.sh
#. ../scripts/build_meta_data_multixlm.sh $sub_tasks $n_samples $add_para_data_to_mono_data

## OR ##
### option 2 : data in a same folder ###
chmod +x ../scripts/build_meta_data_monoxlm.sh
. ../scripts/build_meta_data_monoxlm.sh $sub_tasks $n_samples $add_para_data_to_mono_data

# todo : make things dynamic like this
#chmod +x $PROCESSED_FILE
#$PROCESSED_FILE

echo "======================="
echo "End"
echo "======================="
