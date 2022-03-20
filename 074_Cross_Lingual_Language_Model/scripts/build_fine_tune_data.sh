#!/bin/bash

# Usage : ./opuss.sh $lang_pair

# required :

# All conditions are there to allow the script to resume or it stopped in case of a sudden stop.

pair=$1  # input language pair
N_SAMPLES=${2-'False'}
if [ $N_SAMPLES -le 0 ];then
    N_SAMPLES=False
fi

# 1) if PARA = False && MONO = False : stop and report an error
if [ $PARA = "False" ] && [ $MONO = "False" ]; then
    echo "error"
    exit
# 2) if PARA = False && MONO = False : stop and report an error
elif [ $PARA = "True" ] && [ ! -d $PARA_PATH ]; then
    echo "error"
    exit
# 3) if MONO = True && PARA_PATH does not exist && MONO_PATH does not exist : stop and report an error
elif [ $MONO = "True" ] && [ ! -d $PARA_PATH ] && [ ! -d $MONO_PATH ]; then
    echo "error"
    exit
fi


# 4)
if [ -d $CODE_VOCAB_PATH ]; then
    if [ ! -d $CODE_VOCAB_PATH/vocab ]; then
        echo "file $CODE_VOCAB_PATH/vocab not exists"
        exit
    fi
    if [ ! -d $CODE_VOCAB_PATH/codes ]; then
        echo "file $CODE_VOCAB_PATH/codes not exists"
        exit
    fi
else
    echo "dir $CODE_VOCAB_PATH not exists"
    exit
fi

if [ ! -d $OUTPATH/fine_tune ]; then
    mkdir $OUTPATH/fine_tune
else
    echo "dir $OUTPATH/fine_tune already exists"
fi

# 5) Otherwise, it's okay, we keep going.
echo "params ok !"

#
# Tokenize and preprocess data
#
chmod +x $TOKENIZE

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
      #head -n $NHEAD $1 > $3;
      #tail -n $NTAIL $1 >> $3;
      shuf --random-source=<(get_seeded_random 42) $1 | head $NHEAD   > $3;
      shuf --random-source=<(get_seeded_random 42) $1 | tail $NTAIL   >> $3;
    fi
}

#  para data 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    echo "*** Cleaning and tokenizing $pair data ... ***"
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
            if [ $N_SAMPLES = "False" ];then
                cat $PARA_PATH/$pair.$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
            else
                get_n_samples $PARA_PATH/$pair.$lg.txt $N_SAMPLES $PARA_PATH/samples.$pair.$lg
                cat $PARA_PATH/samples.$pair.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all 
                # todo : memory
                rm $PARA_PATH/${pair}/samples.$pair.$lg
            fi
            echo "*** Tokenized (+ lowercase + accent-removal) $pair.$lg data to $PARA_PATH/? ***"
        else
            #rm $PARA_PATH/$pair.$lg.all
            echo "file $PARA_PATH/$pair.$lg.all already exists"             fi
        fi
    done
fi

# mono data 
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        if [ ! -f $MONO_PATH/$lg.all ]; then
            if [ $N_SAMPLES = "False" ];then
                cat $MONO_PATH/$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all
            else
                get_n_samples $MONO_PATH/$lg.txt $N_SAMPLES $MONO_PATH/samples.$lg
                cat $MONO_PATH/samples.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all 
                # todo : memory
                rm $MONO_PATH/samples.$lg
            fi
            echo "*** Tokenized (+ lowercase + accent-removal) $lg data to $MONO_PATH/? ***"
        else
            #rm $PARA_PATH/$pair.$lg.all
            echo "file $MONO_PATH/$lg.all already exists" 
        fi
    done
fi

# Let's take the case $pair = "en-fr"
# At this point we have for this pair the following files:
# if PARA = True && PARA_PATH exists, in $PARA_PATH: en-en.en.all and en-en.fr.all
# if MONO = True && MONO_PATH exists, in $MONO_PATH: en.all and fr.all

#
# split into train / valid / test
#
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

# para 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test $test_size $val_size
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        split_data $MONO_PATH/$lg.all $MONO_PATH/$lg.train $MONO_PATH/$lg.valid $MONO_PATH/$lg.test $test_size $val_size
    done
fi

# Let's take the case $pair = "en-fr"
# At this point we have, in addition to the previous files, the following files:
# if PARA = True && PARA_PATH exists, in $PARA_PATH: en-fr.en.train and en-fr.fr.train, en-fr.en.valid and
#                                                    en-fr.fr.valid, en-fr.en.test and en-fr.fr.test
# if MONO = True && MONO_PATH exists, in $MONO_PATH: en.train and fr.train, en.valid and fr.valid, en.test et fr.test

#
# Now we create our training set for the BPE vocabulary, for instance by taking 100M sentences from each 
# monolingua corpora.
# 
  
echo -e "\n"
echo "***Apply BPE tokenization on the corpora and binarize everything using preprocess.py.***"

# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        for split in train valid test; do
            $FASTBPE applybpe $OUTPATH/fine_tune/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $CODE_VOCAB_PATH/codes
            python preprocess.py $CODE_VOCAB_PATH/vocab $OUTPATH/fine_tune/$pair.$lg.$split
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        for split in train valid test; do
            $FASTBPE applybpe $OUTPATH/fine_tune/$split.$lg $MONO_PATH/$lg.$split $CODE_VOCAB_PATH/codes
            # Add para data to mono data before preprocessing
            if [ $PARA = "True" ]; then
                for lg_tmp in $(echo $pair | sed -e 's/\-/ /g'); do
                    for split_tmp in train valid test; do
                        # Add the contents of $OUTPATH/$pair.$lg_tmp.$split_tmp after $OUTPATH/$split.$lg
                        cat $OUTPATH/fine_tune/$pair.$lg_tmp.$split_tmp >> $OUTPATH/$split.$lg
                    done
                done
            fi
            python preprocess.py $CODE_VOCAB_PATH/vocab $OUTPATH/fine_tune/$split.$lg
        done
    done
fi

# if MONO = True && MONO_PATH does not exist && PARA_PATH exists
if [ $MONO = "True" ] && [ ! -d $MONO_PATH ] && [ -d $PARA_PATH ]; then
    # We use our parallel data to construct the monolingual data 
    echo -e "\n"
    echo "***Using parallel data to construct monolingual data***"
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        for split in train valid test; do
            cp $OUTPATH/fine_tune/$pair.$lg.$split.pth $OUTPATH/fine_tune/$split.$lg.pth      
        done
    done
fi

echo -e "\n"
echo "***Creat the file to train the XLM model with MLM+TLM objective***"

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    for split in train valid test; do
        cp $OUTPATH/fine_tune/$pair.$lg.$split.pth $OUTPATH/fine_tune/$split.$pair.$lg.pth  
    done
done


echo -e "\n"
echo "*** build data with succes : dir $OUTPATH/fine_tune ***"