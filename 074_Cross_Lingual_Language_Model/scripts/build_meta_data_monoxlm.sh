#!/bin/bash

# Usage: ./build_meta_data.sh $sub_task $n_samples $sub_task_data

# All conditions are there to allow the script to resume or it stopped in case of a sudden stop.

set -e

# 1) if no parameters : stop
if [ $# = 0 ];then
  exit
fi

SUB_TASKS_DATA_PERCENT=${1-""}
# 2) if no task : stop
if [ $SUB_TASKS_DATA_PERCENT = "" ];then
    exit
fi

N_SAMPLES=${2-'False'}
if [ $N_SAMPLES -le 0 ];then
    N_SAMPLES=False
fi

sub_tasks=""
fine_tune_data_percent=""

# For each task we have $language_pair:fine_tune_data_percent. We separate the two.
for task_data_percent in $(echo $SUB_TASKS_DATA_PERCENT | sed -e 's/\,/ /g'); do
    IFS=': ' read -r -a array <<< "$task_data_percent"
    sub_tasks=$sub_tasks,${array[0]}
    fine_tune_data_percent=$fine_tune_data_percent,${array[1]}
done

# Remove the comma in front
sub_tasks=$(echo $sub_tasks | cut -c2-)
fine_tune_data_percent=$(echo $fine_tune_data_percent | cut -c2-)

# 4) if no task : stop
if [ $sub_tasks = "" ];then
    exit
fi

if [ $fine_tune_data_percent != "" ];then
    if [ ! -d $OUTPATH/fine_tune ]; then
        mkdir $OUTPATH/fine_tune
    else
        echo "dir $OUTPATH/fine_tune already exists"
    fi
fi

# 5) if PARA = False && MONO = False : stop and report an error
if [ $PARA = "False" ] && [ $MONO = "False" ]; then
    echo "error"
    exit
# 6) if PARA = False && MONO = False : stop and report an error
elif [ $PARA = "True" ] && [ ! -d $PARA_PATH ]; then
    echo "error"
    exit
# 7) if MONO = True && PARA_PATH does not exist && MONO_PATH does not exist : stop and report an error
elif [ $MONO = "True" ] && [ ! -d $PARA_PATH ] && [ ! -d $MONO_PATH ]; then
    echo "error"
    exit
fi

# 8) Otherwise, it's okay, we keep going.
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
      head -n $NHEAD $1 > $3;
      tail -n $NTAIL $1 >> $3;
      #shuf --random-source=<(get_seeded_random 42) $1 | head $NHEAD   > $3;
      #shuf --random-source=<(get_seeded_random 42) $1 | tail $NTAIL   >> $3;
    fi
}

#  para data 
# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        echo "*** Cleaning and tokenizing $pair data ... ***"
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
                if [ $N_SAMPLES = "False" ];then
                    cat $PARA_PATH/$pair.$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
                else
                    get_n_samples $PARA_PATH/$pair.$lg.txt $N_SAMPLES $PARA_PATH/samples.$pair.$lg
                    cat $PARA_PATH/samples.$pair.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all 
                    # todo : memory
                    rm $PARA_PATH/samples.$pair.$lg
                fi
                echo "*** Tokenized (+ lowercase + accent-removal) $pair.$lg data to $PARA_PATH/? ***"
            else
                #rm $PARA_PATH/$pair.$lg.all
                echo "file $PARA_PATH/$pair.$lg.all already exists" 
            fi
        done
    done
fi

# mono data 
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
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
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test $test_size $val_size
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            split_data $MONO_PATH/$lg.all $MONO_PATH/$lg.train $MONO_PATH/$lg.valid $MONO_PATH/$lg.test $test_size $val_size
        done
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

echo -e "\n\n"
echo "***build the training set for BPE tokenization ($nCodes codes)***"

# I'm only handling the case SAME_VOCAB = True for now

echo -e "\n"
echo "***shuf ... Generating $shuf_n_samples random permutations of training data and store result in $OUTPATH/${pair}/bpe.train***"

# para 
# if PARA = True (then PARA_PATH must exist)

if [ ! -f $OUTPATH/bpe.train ]; then
    if [ $PARA = "True" ]; then
        for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
            for lg in $(echo $pair | sed -e 's/\-/ /g'); do
                shuf -r -n $shuf_n_samples $PARA_PATH/$pair.$lg.train >> $OUTPATH/bpe.train
            done
        done
    fi

    # mono
    # if MONO = True &&  MONO_PATH exist
    if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
        for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
            for lg in $(echo $pair | sed -e 's/\-/ /g'); do
                shuf -r -n $shuf_n_samples $MONO_PATH/$lg.train >> $OUTPATH/bpe.train
            done
        done
    fi
else
    #rm $OUTPATH/bpe.train
    echo "file $OUTPATH/bpe.train already exists"
fi

echo -e "\n"
echo "***Learn the BPE vocabulary on the training set : $OUTPATH/bpe.train ...***"
if [ ! -f $OUTPATH/codes ]; then
     $FASTBPE learnbpe $nCodes $OUTPATH/bpe.train > $OUTPATH/codes
else
    #rm $OUTPATH/codes
    echo "file $OUTPATH/codes already exists"
fi

echo "***Learn $nCodes BPE code on the bpe.train file***" 

echo -e "\n"
echo "***Get the post-BPE vocab***"
if [ ! -f $OUTPATH/train ]; then
    $FASTBPE applybpe $OUTPATH/train $OUTPATH/bpe.train $OUTPATH/codes
else
    #rm $OUTPATH/train
    echo "file $OUTPATH/train already exists"
fi

if [ ! -f $OUTPATH/vocab ]; then
    cat $OUTPATH/train | $FASTBPE getvocab - > $OUTPATH/vocab
else
    #rm $OUTPATH/vocab
    echo "file $OUTPATH/vocab already exists"
fi
  
echo -e "\n"
echo "***Apply BPE tokenization on the corpora.***"

# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                if [ ! -f $OUTPATH/$pair.$lg.$split ]; then
                    $FASTBPE applybpe $OUTPATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $OUTPATH/codes
                else
                    echo "file $OUTPATH/$pair.$lg.$split already exists"
                fi
            done
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                if [ ! -f $OUTPATH/$split.$lg ]; then
                    $FASTBPE applybpe $OUTPATH/$split.$lg $MONO_PATH/$lg.$split $OUTPATH/codes
                    # Add para data to mono data before preprocessing
                    add_para_data_to_mono_data=${3-'True'}
                    if [ $add_para_data_to_mono_data = "True" ]; then 
                      if [ $PARA = "True" ]; then
                          for lg_tmp in $(echo $pair | sed -e 's/\-/ /g'); do
                              for split_tmp in train valid test; do
                                  # Add the contents of $OUTPATH/$pair.$lg_tmp.$split_tmp after $OUTPATH/$split.$lg
                                  cat $OUTPATH/$pair.$lg_tmp.$split_tmp >> $OUTPATH/$split.$lg
                              done
                          done
                      fi
                    fi
                else
                    echo "file $OUTPATH/$split.$lg already exists"
                fi
            done
        done
    done
fi

echo -e "\n"
echo "***Build fine_tune data***"

# Usage : build_fine_tune_data $sub_tasks $fine_tune_data_percent 
build_fine_tune_data() {
    get_seeded_random() {
        seed="$1"; openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
    };
    
    IFS=', ' read -r -a array1 <<< "$1"
    IFS=', ' read -r -a array2 <<< "$2"
    
    #echo ${array1[*]}
    #echo ${array2[*]}
    
    for (( i=0; i<${#array1[*]}; ++i)); do 
        data_percent=${array2[$i]}
        if [ $data_percent != "" ] && [ $data_percent -gt 0 ] ;then
            pair=${array1[$i]}
            for lg in $(echo $pair | sed -e 's/\-/ /g'); do
                for split in train valid test; do
                    # PARA
                    if [ $PARA = "True" ]; then
                        name=$OUTPATH/$pair.$lg.$split
                        NLINES=`wc -l $name`;
                        IFS=' ' read -r -a array <<< "$NLINES"
                        NLINES=${array[0]}
                        NLINES=$(($NLINES+1));
                        N_FINE_TUNE=$((($NLINES*$data_percent)/100))
                        if [ $NLINES -le $N_FINE_TUNE ]; then
                            # todo : exit
                            echo "error"
                        else
                            NREST=$(($NLINES - $N_FINE_TUNE));
                            mv $OUTPATH/$pair.$lg.$split $OUTPATH/$pair.$lg.$split.tmp
                            shuf --random-source=<(get_seeded_random 42) $OUTPATH/$pair.$lg.$split.tmp | head -$NREST > $OUTPATH/$pair.$lg.$split;
                            shuf --random-source=<(get_seeded_random 42) $OUTPATH/$pair.$lg.$split.tmp | tail -$N_FINE_TUNE > $OUTPATH/fine_tune/$pair.$lg.$split;
                            # todo : memory
                            rm $OUTPATH/$pair.$lg.$split.tmp
                        fi
                    fi

                    # MONO
                    if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
                        name=$OUTPATH/$split.$lg
                        #NLINES=`wc -l $name | awk -F " " '{print $name}'`;
                        NLINES=`wc -l $name`;
                        IFS=' ' read -r -a array <<< "$NLINES"
                        NLINES=${array[0]}
                        NLINES=$(($NLINES+1));
                        N_FINE_TUNE=$((($NLINES*$data_percent)/100))
                        if [ $NLINES -le $N_FINE_TUNE ]; then
                            # todo : exit
                            echo "error"
                        else
                            NREST=$(($NLINES - $N_FINE_TUNE));
                            mv $OUTPATH/$split.$lg $OUTPATH/$split.$lg.tmp
                            shuf --random-source=<(get_seeded_random 42) $OUTPATH/$split.$lg.tmp | head -$NREST > $OUTPATH/$split.$lg; 
                            shuf --random-source=<(get_seeded_random 42) $OUTPATH/$split.$lg.tmp | tail -$N_FINE_TUNE > $OUTPATH/fine_tune/$split.$lg;
                            # todo : memory
                            rm $OUTPATH/$split.$lg.tmp
                        fi
                    fi
                done
            done
        fi
    done
}

build_fine_tune_data $sub_tasks $fine_tune_data_percent

echo -e "\n"
echo "***Binarize everything using preprocess.py.***"

# if PARA = True (then PARA_PATH must exist)
if [ $PARA = "True" ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                if [ ! -f $OUTPATH/$pair.$lg.$split.pth ]; then
                    python preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
                else
                    echo "file $OUTPATH/$pair.$lg.$split.pth already exists"
                fi
                if [ -f $OUTPATH/fine_tune/$pair.$lg.$split ]; then
                    if [ ! -f $OUTPATH/fine_tune/$pair.$lg.$split.pth ]; then
                        python preprocess.py $OUTPATH/vocab $OUTPATH/fine_tune/$pair.$lg.$split
                    else
                        echo "file $OUTPATH/fine_tune/$pair.$lg.$split.pth already exists"
                    fi
                fi
            done
        done
    done
fi

# mono
# if MONO = True &&  MONO_PATH exist
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                if [ ! -f $OUTPATH/$split.$lg.pth ]; then
                    python preprocess.py $OUTPATH/vocab $OUTPATH/$split.$lg
                else
                    echo "file $OUTPATH/$split.$lg.pth already exists"
                fi
                if [ -f $OUTPATH/fine_tune/$split.$lg ]; then
                    if [ ! -f $OUTPATH/fine_tune/$split.$lg.pth ]; then
                        python preprocess.py $OUTPATH/vocab $OUTPATH/fine_tune/$split.$lg
                    else
                        echo "file $OUTPATH/fine_tune/$split.$lg.pth already exists"
                    fi
                fi
            done
        done
    done
fi


# if MONO = True && MONO_PATH does not exist && PARA_PATH exists
if [ $MONO = "True" ] && [ ! -d $MONO_PATH ] && [ -d $PARA_PATH ]; then
    # We use our parallel data to construct the monolingual data 
    echo -e "\n"
    echo "***Using parallel data to construct monolingual data***"
    for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
            for split in train valid test; do
                if [ ! -f $OUTPATH/$split.$lg.pth ]; then
                    cp $OUTPATH/$pair.$lg.$split.pth $OUTPATH/$split.$lg.pth
                else
                    echo "file $OUTPATH/$split.$lg.pth already exists"
                fi
                if [ -f $OUTPATH/fine_tune/$pair.$lg.$split.pth ]; then
                    if [ ! -f $OUTPATH/fine_tune/$split.$lg.pth ]; then
                        cp $OUTPATH/fine_tune/$pair.$lg.$split.pth $OUTPATH/fine_tune/$split.$lg.pth
                    else
                        echo "file $OUTPATH/fine_tune/$split.$lg.pth already exists"
                    fi
                fi  
            done
        done
    done
fi

echo -e "\n"
echo "***Creat the file to train the XLM model with MLM+TLM objective***"
for pair in $(echo $sub_tasks | sed -e 's/\,/ /g'); do
    for lg in $(echo $pair | sed -e 's/\-/ /g'); do
        for split in train valid test; do
            if [ ! -f $OUTPATH/$split.$pair.$lg.pth ]; then
                cp $OUTPATH/$pair.$lg.$split.pth $OUTPATH/$split.$pair.$lg.pth
            else
                echo "file $OUTPATH/$split.$pair.$lg.pth already exists"
            fi
            if [ -f $OUTPATH/fine_tune/$pair.$lg.$split.pth ]; then
                if [ ! -f $OUTPATH/fine_tune/$split.$pair.$lg.pth ]; then
                    cp $OUTPATH/fine_tune/$pair.$lg.$split.pth $OUTPATH/fine_tune/$split.$pair.$lg.pth
                else
                    echo "file $OUTPATH/fine_tune/$split.$pair.$lg.pth already exists"
                fi
            fi    
        done
    done
done

echo -e "\n"
echo "*** build data with succes : dir $OUTPATH ***"
