#!/bin/bash

# Usage: ./get-data-para.sh $lg_pair \
#                          --PARA False # Si on doit traiter les données para
#                          --MONO False # Si on doit traiter les données mono
#                          --PARA_PATH ... # dossier contenant les données para
#                          --MONO_PATH ... # dossier contenant les données mono
#                          --SAME_VOCAB True # si les deux langues doivent partager le même vocabulaire
#                          --XLM_PATH ... # dossier de clonage de XLM
#                          --PROCESSED_PATH ... # dossier dans lequel seront stockées les données issues du traitement
#                          --nCodes 50000
#                          --shuf_n_samples 10000000

# see get_data_and_train_model.ipynb notebook in Colab notebook dir

set -e

pair=$1  # input language pair

# 1) si PARA = False && MONO = False : on arrete et on signale une erreur
if [ $PARA = "False" ] && [ $MONO = "False" ]; then
    echo "error : Je m'arrete ici! 1"
    exit
# 2) si PARA = True && PARA_PATH n'existe pas : on arrete et on signale une erreur
elif [ $PARA = "True" ] && [ ! -d $PARA_PATH ]; then
    echo "error : Je m'arrete ici! 2"
    exit
# 3) si MONO = True && PARA_PATH n'existe pas && MONO_PATH n'existe pas : on arrete et on signale une erreur
elif [ $MONO = "True" ] && [ ! -d $PARA_PATH ] && [ ! -d $MONO_PATH ]; then
    echo "error : Je m'arrete ici! 3"
    exit
fi
# 4) sinon, c'est ok, on continue
echo "params ok !"

#
# Tokenize and preprocess data
#
chmod +x $TOKENIZE
#  para data 
# si PARA = True (alors PARA_PATH existe  forcement)
if [ $PARA = "True" ]; then
  echo "*** Cleaning and tokenizing $pair data ... ***"
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
      if [ ! -f $PARA_PATH/$pair.$lg.all ]; then
        cat $PARA_PATH/$pair.$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
        #cat $PARA_PATH/$pair.$lg.* | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $PARA_PATH/$pair.$lg.all
        echo "*** Tokenized (+ lowercase + accent-removal) $pair.$lg data to $PARA_PATH/? ***"
      else
        #rm $PARA_PATH/$pair.$lg.all
        echo "error, le fichier $PARA_PATH/$pair.$lg.all existe deja" 
      fi
  done
fi

# mono data 
# si MONO = True &&  MONO_PATH existe
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    if [ ! -f $MONO_PATH/$lg.all ]; then
        #cat $MONO_PATH/$lg.txt | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all
        cat $MONO_PATH/$lg.* | $TOKENIZE $lg | python $LOWER_REMOVE_ACCENT > $MONO_PATH/$lg.all
        echo "*** Tokenized (+ lowercase + accent-removal) $lg data to $MONO_PATH/? ***"
    else
        #rm $PARA_PATH/$pair.$lg.all
        echo "error, le fichier $MONO_PATH/$lg.all existe deja" 
    fi
  done
fi

# Supposons que $pair = "en-fr"
# A ce stade on à les fichiers suivants :
# si PARA = True && PARA_PATH existe, dans $PARA_PATH : en-fr.en.all et en-fr.fr.all
# si MONO = True && MONO_PATH existe, dans $MONO_PATH : en.all et fr.all

#
# split into train / valid / test
#
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

# para 
# si PARA = True (alors PARA_PATH existe  forcement)
if [ $PARA = "True" ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    split_data $PARA_PATH/$pair.$lg.all $PARA_PATH/$pair.$lg.train $PARA_PATH/$pair.$lg.valid $PARA_PATH/$pair.$lg.test $test_size $val_size
  done
fi

# mono
# si MONO = True &&  MONO_PATH existe
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    split_data $MONO_PATH/$lg.all $MONO_PATH/$lg.train $MONO_PATH/$lg.valid $MONO_PATH/$lg.test $test_size $val_size
  done
fi

# Supposons que $pair = "en-fr"
# A ce stade on a, en plus des fichiers précedents, les fichiers suivants :
# si PARA = True && PARA_PATH existe, dans $PARA_PATH : en-fr.en.train et en-fr.fr.train, en-fr.en.valid et
#                           en-fr.fr.valid, en-fr.en.test et en-fr.fr.test
# si MONO = True && MONO_PATH existe, dans $MONO_PATH : en.train et fr.train, en.valid et fr.valid, 
#                                   en.test et fr.test

#
# Now we create our training set for the BPE vocabulary, for instance by taking 100M sentences from each 
# monolingua corpora.
# 

echo "***build the training set for BPE tokenization ($nCodes codes)***"

# Je ne gère que le cas SAME_VOCAB = True pour l'instant

# para 
# si PARA = True (alors PARA_PATH existe  forcement)
if [ $PARA = "True" ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    shuf -r -n $shuf_n_samples $PARA_PATH/$pair.$lg.train >> $OUTPATH/bpe.train
  done
fi

# mono
# si MONO = True &&  MONO_PATH existe
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    shuf -r -n $shuf_n_samples $MONO_PATH/$lg.train >> $OUTPATH/bpe.train
  done
fi

echo "***Learn the BPE vocabulary on the training set : $OUTPATH/bpe.train***"
$FASTBPE learnbpe $nCodes $OUTPATH/bpe.train > $OUTPATH/codes

echo "***Get the post-BPE vocab***"
$FASTBPE applybpe $OUTPATH/train $OUTPATH/bpe.train $OUTPATH/codes 
cat $OUTPATH/train | $FASTBPE getvocab - > $OUTPATH/vocab 
echo "***Learn the $nCodes BPE code on the bpe.train file***" 

echo "***Apply BPE tokenization on the monolingual and parallel corpora, and binarize everything using preprocess.py.***"
# si PARA = True (alors PARA_PATH existe  forcement)
if [ $PARA = "True" ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    for split in train valid test; do
      $FASTBPE applybpe $OUTPATH/$pair.$lg.$split $PARA_PATH/$pair.$lg.$split $OUTPATH/codes
      python preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
    done
  done
fi

# mono
# si MONO = True &&  MONO_PATH existe
if [ $MONO = "True" ] && [ -d $MONO_PATH ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    for split in train valid test; do
      $FASTBPE applybpe $OUTPATH/$split.$lg $MONO_PATH/$lg.$split $OUTPATH/codes
      # Ajouter les données para au données mono avant le preprocess
      if [ $PARA = "True" ]; then
        for lg in $(echo $pair | sed -e 's/\-/ /g'); do
          for split in train valid test; do
            # Ajouter le contenu de $OUTPATH/$pair.$lg.$split à la suite de $OUTPATH/$split.$lg
            cat $OUTPATH/$pair.$lg.$split >> $OUTPATH/$split.$lg
          done
        done
      fi
      python preprocess.py $OUTPATH/vocab $OUTPATH/$split.$lg
    done
  done
fi

# si MONO = True && MONO_PATH n'existe pas && PARA_PATH existe
if [ $MONO = "True" ] && [ ! -d $MONO_PATH ] && [ ! -d $PARA_PATH ]; then
  # On utilise nos para data pour les mono
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    for split in train valid test; do
      cp $OUTPATH/$pair.$lg.$split.pth $OUTPATH/$split.$lg.pth
    done
  done
fi