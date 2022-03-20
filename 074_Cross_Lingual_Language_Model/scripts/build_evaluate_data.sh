#!/bin/bash

# Usage: ./build_evaluate_data.sh $lg $n_samples 

# required :
#   src_path 
#   tgt_path
#   TOKENIZE 
#   threads_for_tokenizer
#   LOWER_REMOVE_ACCENT 
#   test_size 
#   val_size
#   FASTBPE 
#   CODE_VOCAB_PATH
#   duplicate 

set -e

if [ $val_size=0 ];then
    duplicate=True
else
    duplicate=False
fi

# tools paths
TOOLS_PATH=tools
TOKENIZE=$TOOLS_PATH/tokenizer_our.sh
LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
FASTBPE=$TOOLS_PATH/fastBPE/fast

chmod +x $FASTBPE
chmod +x $TOOLS_PATH/mosesdecoder/scripts/tokenizer/*.perl

lg=$1
N_SAMPLES=${2-'False'}
if [ $N_SAMPLES -le 0 ];then
    N_SAMPLES=False
fi

# usage : get_n_samples input_file n_samples output_file
get_n_samples() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NLINES=$(($NLINES+1));
    if [ $NLINES -le $2 ]; then
      cp $1 $3
    else
      NTAIL=$(($2/2));
      NHEAD=$(($2 - $NTAIL));
      head -n $NHEAD $1 > $3;
      tail -n $NTAIL $1 >> $3;
      #shuf --random-source=<(get_seeded_random 42) $1 | head $NHEAD   > $3;
      #shuf --random-source=<(get_seeded_random 42) $1 | tail $NTAIL   >> $3;
    fi
}

echo "*** Cleaning and tokenizing $lg data ... ***"
if [ ! -f $src_path/$lg.all ]; then
    if [ $N_SAMPLES = "False" ];then
        cat $src_path/$lg.txt | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $src_path/$lg.all
    else
        get_n_samples $src_path/$lg.txt $N_SAMPLES $src_path/samples.$lg.txt 
        cat $src_path/samples.$lg.txt | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $src_path/$lg.all
        # todo : memory
        rm $src_path/samples.$lg.txt
    fi
    echo "*** Tokenized (+ lowercase + accent-removal) $lg data to $src_path/$lg.all ***"
else
    #rm $src_path/$lg.all
    echo "file $src_path/$lg.all already exists" 
fi

echo -e "\n"
echo "*** split into train / valid / test ***"
split_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NLINES=$(($NLINES+1));
    NTEST=$(((NLINES*$5)/100));
    NVAL=$(((NLINES*$6)/100));
    NTRAIN=$((NLINES - $NVAL - $NTEST));
    # todo : correct this error. But the code works with it.
    # shuf: write error
    # shuf: write error: Broken pipe
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                           > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$(($NTRAIN+$NVAL)) | tail -$NVAL  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$NTEST                            > $4;
}

if [ ! -d $tgt_path ]; then
    mkdir $tgt_path
fi
    
if [ $duplicate = True ]; then
    $FASTBPE applybpe $tgt_path/test.$lg $src_path/$lg.all $CODE_VOCAB_PATH/codes 
    python preprocess.py $CODE_VOCAB_PATH/vocab $tgt_path/test.$lg
    cp $tgt_path/test.$lg.pth $tgt_path/valid.$lg.pth
else
    if [ ! -f $src_path/train.$lg ]; then
        split_data $src_path/$lg.all $src_path/train.$lg $src_path/valid.$lg $src_path/test.$lg $test_size $val_size
    fi

    echo -e "\n"
    echo "***Apply BPE tokenization on the corpora and binarize everything using preprocess.py.***"
    #for split in train valid test; do
    for split in valid test; do
       $FASTBPE applybpe $tgt_path/$split.$lg $src_path/$split.$lg $CODE_VOCAB_PATH/codes 
       python preprocess.py $CODE_VOCAB_PATH/vocab $tgt_path/$split.$lg
    done
fi

