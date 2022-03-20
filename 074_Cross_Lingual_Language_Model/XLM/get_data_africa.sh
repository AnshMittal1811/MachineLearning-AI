#!/bin/bash
# Usage: ./get_data_hypothesis.sh $lg_pair \
#                          --PARA_PATH ... # dossier (où stocker/contenant) les données para
#                          --nCodes
#                          --shuf_n_samples 
#                          --threads_for_tokenizer
#                          --...
#                          --...
#                          --...

# Tout les conditions sont la pour permettre au script de reprendre ou il s'est arrété en cas d'arret brusque

set -e

#########
# dossier (contenant/dans le quel stocker) les données 
PARA_PATH=data/africa/para
MONO_PATH=data/africa/mono

nCodes=300
shuf_n_samples=1000

# Pourcentage des données de test et de validation (en %) 
# todo : bon ? 
test_size=10
val_size=10

# tools paths
#% env TOKENIZE=$TOOLS_PATH/tokenize.sh
TOKENIZE=tools/tokenizer_our.sh
#% env LOWER_REMOVE_ACCENT=$TOOLS_PATH/lowercase_and_remove_accent.py
LOWER_REMOVE_ACCENT=tools/lowercase_and_remove_accent.py
#% env FASTBPE=$TOOLS_PATH/fastBPE/fast
FASTBPE=tools/fastBPE/fast

#% env OUTPATH $PROCESSED_PATH/$pair/$nCodes
# nommage du fichier en fonction de 'nCodes'
OUTPATH=data/africa/processed/30000
sudo mkdir -p $OUTPATH

sudo chmod +x $FASTBPE
sudo chmod +x get_data_hypothesis.sh
sudo chmod +x tools/mosesdecoder/scripts/tokenizer/*.perl

threads_for_tokenizer=16
#########


pair=$1  # input language pair

if [ ! -d $PARA_PATH/${pair} ]; then
  sudo mkdir $PARA_PATH/${pair}
else
  echo "dir $PARA_PATH/${pair} already exists"
fi

if [ ! -d $OUTPATH/${pair} ]; then
  sudo mkdir $OUTPATH/${pair}
else
  echo "dir $OUTPATH/${pair} already exists"
fi

#
# Tokenize and preprocess data
#
chmod +x $TOKENIZE

# usage : get_n_samples input_file n_samples output_file
get_n_samples() {
    NLINES=`wc -l $1  | awk -F " " '{print $1}'`;
    NLINES=$(($NLINES+1));
    if [ $NLINES -le $2 ]; then
      cp $1 $3
    else
      NTAIL=$(($2/2));
      NHEAD=$(($2 - NTAIL));
      head -n $NHEAD $1 > $3;
      tail -n $NTAIL $1 >> $3;
    fi
}

N_SAMPLES=${2-'false'}

# tokenize
echo -e "\n"
echo "*** Cleaning and tokenizing $pair data ... ***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/${pair}/$pair.$lg.all ]; then
    if [ $N_SAMPLES = "false" ];then
      if [ $lg = "Anglais" ] || [ $lg = "Francais" ]; then 
        cat $PARA_PATH/${pair}/$pair.$lg.txt | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/${pair}/$pair.$lg.all
      else
        # On ne fait pas de lower case et remove accent sur les langues africaines
        cat $PARA_PATH/${pair}/$pair.$lg.txt | $TOKENIZE $lg $threads_for_tokenizer > $PARA_PATH/${pair}/$pair.$lg.all
      fi
      # Si les données monolingues sont disponibles pour une langue donnée
      if [ -d $MONO_PATH/$lg ]; then
        if [ ! -f $MONO_PATH/$lg/$lg.all ]; then
          if [ $lg = "Anglais" ] || [ $lg = "Francais" ]; then 
            cat $MONO_PATH/$lg/$lg.txt | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg/$lg.all
          else
            # On ne fait pas de lower case et remove accent sur les langues africaines
            cat $MONO_PATH/$lg/$lg.txt | $TOKENIZE $lg $threads_for_tokenizer > $MONO_PATH/$lg/$lg.all
          fi
        else
          echo "file $MONO_PATH/$lg/$lg.all already exists"
        fi
      fi
    else
      cat $PARA_PATH/${pair}/$pair.$lg.txt > $PARA_PATH/${pair}/all.$pair.$lg
      get_n_samples $PARA_PATH/${pair}/all.$pair.$lg $N_SAMPLES $PARA_PATH/${pair}/samples.$pair.$lg
      cat $PARA_PATH/${pair}/samples.$pair.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/${pair}/$pair.$lg.all
      rm $PARA_PATH/${pair}/all.$pair.$lg
      rm $PARA_PATH/${pair}/samples.$pair.$lg
    fi
    echo "*** Tokenized (+ lowercase + accent-removal) $pair.$lg data to $PARA_PATH/${pair}/? ***"
  else
    #rm $PARA_PATH/${pair}/$pair.$lg.all
    echo "file $PARA_PATH/$pair/$pair.$lg.all already exists"
  fi
done

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
    shuf --random-source=<(get_seeded_random 42) $1 | head -$NTRAIN                           > $2;
    shuf --random-source=<(get_seeded_random 42) $1 | head -$(($NTRAIN+$NVAL)) | tail -$NVAL  > $3;
    shuf --random-source=<(get_seeded_random 42) $1 | tail -$NTEST                            > $4;
}

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  split_data $PARA_PATH/${pair}/$pair.$lg.all $PARA_PATH/${pair}/$pair.$lg.train $PARA_PATH/${pair}/$pair.$lg.valid $PARA_PATH/${pair}/$pair.$lg.test $test_size $val_size
  if [ -f $MONO_PATH/$lg/$lg.all ]; then
    split_data $MONO_PATH/$lg/$lg.all $MONO_PATH/$lg/$lg.train $MONO_PATH/$lg/$lg.valid $MONO_PATH/$lg/$lg.test $test_size $val_size 
  fi
done

echo -e "\n\n"
echo "***build the training set for BPE tokenization ($nCodes codes)***"

echo -e "\n"
echo "***shuf ... Generating $shuf_n_samples random permutations of train data and store result in $OUTPATH/${pair}/bpe.train***"
if [ ! -f $OUTPATH/${pair}/bpe.train ]; then
  sudo touch $OUTPATH/${pair}/bpe.train
  sudo chmod a+w $OUTPATH/${pair}/bpe.train
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    shuf -r -n $shuf_n_samples $PARA_PATH/${pair}/$pair.$lg.train >> $OUTPATH/${pair}/bpe.train
    if [ -f $MONO_PATH/$lg/$lg.train ]; then
      shuf -r -n $shuf_n_samples $MONO_PATH/$lg/$lg.train >> $OUTPATH/${pair}/bpe.train 
    fi
  done
else
  #rm $OUTPATH/${pair}/bpe.train
  echo "file $OUTPATH/${pair}/bpe.train already exists"
fi

echo -e "\n"
echo "***Learn the BPE vocabulary on the training set : $OUTPATH/bpe.train***"
if [ ! -f $OUTPATH/${pair}/codes ]; then
  sudo touch $OUTPATH/${pair}/codes
  sudo chmod a+w $OUTPATH/${pair}/codes
  #sudo $FASTBPE learnbpe $nCodes $OUTPATH/${pair}/bpe.train >> $OUTPATH/${pair}/codes
  sudo $FASTBPE learnbpe 10 $OUTPATH/${pair}/bpe.train >> $OUTPATH/${pair}/codes
else
  #rm $OUTPATH/${pair}/codes
  echo "file $OUTPATH/${pair}/codes already exists"
fi

echo -e "\n"
echo "***Get the post-BPE vocab***"
if [ ! -f $OUTPATH/${pair}/train ]; then
  $FASTBPE applybpe $OUTPATH/${pair}/train $OUTPATH/${pair}/bpe.train $OUTPATH/${pair}/codes 
else
  #rm $OUTPATH/${pair}/train
  echo "file $OUTPATH/${pair}/train already exists"
fi
if [ ! -f $OUTPATH/${pair}/vocab ]; then
  cat $OUTPATH/${pair}/train | $FASTBPE getvocab - > $OUTPATH/${pair}/vocab 
else
  #rm $OUTPATH/${pair}/vocab
  echo "file $OUTPATH/${pair}/vocab already exists"
fi

echo -e "\n"
echo "***Apply BPE tokenization on the parallel corpora, and binarize everything using preprocess.py.***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/${pair}/$pair.$lg.$split $PARA_PATH/${pair}/$pair.$lg.$split $OUTPATH/${pair}/codes
    python preprocess.py $OUTPATH/${pair}/vocab $OUTPATH/${pair}/$pair.$lg.$split
    if [ -f $MONO_PATH/$lg/$lg.$split ]; then
      $FASTBPE applybpe $OUTPATH/${pair}/$lg.$split $MONO_PATH/$lg/$lg.$split $OUTPATH/${pair}/codes
      python preprocess.py $OUTPATH/${pair}/vocab $OUTPATH/${pair}/$lg.$split
    fi
  done
done

echo -e "\n"
echo "***Using parallel data to construct monolingual data***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    if [ ! -f $OUTPATH/${pair}/$split.$lg.pth ]; then
      cp $OUTPATH/${pair}/$pair.$lg.$split.pth $OUTPATH/${pair}/$split.$lg.pth
    fi
  done
done

echo -e "\n"
echo "***Creat the file to train the XLM model with MLM+TLM objective***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    cp $OUTPATH/${pair}/$pair.$lg.$split.pth $OUTPATH/${pair}/$split.$pair.$lg.pth
  done
done

echo -e "\n"
echo "*** get data for hypothesis with succes : dir $OUTPATH/${pair} ***"