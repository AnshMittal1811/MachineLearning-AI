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

pair=$1  # input language pair

if [ ! -d $PARA_PATH/${pair} ]; then
  mkdir $PARA_PATH/${pair}
else
  echo "dir $PARA_PATH/${pair} already exists"
fi

if [ ! -d $OUTPATH/${pair} ]; then
  mkdir $OUTPATH/${pair}
else
  echo "dir $PARA_PATH/${pair} already exists"
fi

echo -e "\n"
echo "***Download data and unzip it in $PARA_PATH/$pair ***"

# Take 2 parameters : PARA_PATH source 
download_and_unzip_data() {
  wget -c http://opus.nlpl.eu/download.php?f=$2%2F${pair}.txt.zip -P $1/${pair}
  unzip -u $1/${pair}/download.php?f=$2%2F${pair}.txt.zip -d $1/${pair}
}

if [ $SRC = "MultiUN,OpenSubtitles2018" ] || [ $SRC = "OpenSubtitles2018,MultiUN" ]; then
  if [ $pair != "es-it" ]; then
    # es-fr, de-en, fr-ru, en-ru, de-fr
    for src in $(echo $SRC | sed -e 's/\,/ /g'); do
      download_and_unzip_data $PARA_PATH $src 
    done
  else
    # es-it
    for src in OpenSubtitles GlobalVoices EUbookshop; do
      wget -c http://opus.nlpl.eu/download.php?f=${src}%2Fes-it.txt.zip -P $PARA_PATH/${pair}
      unzip -u $PARA_PATH/${pair}/download.php?f=${src}%2Fes-it.txt.zip -d $PARA_PATH/${pair}
    done
  fi
elif [ $SRC = "MultiUN" ] || [ $SRC = "OpenSubtitles2018" ]; then
  if [ $pair != "es-it" ]; then
    # es-fr, de-en, fr-ru, en-ru, de-fr
    download_and_unzip_data $PARA_PATH $SRC 
  else
    # es-it
    wget -c http://opus.nlpl.eu/download.php?f=OpenSubtitles2018%2Fes-it.txt.zip -P $PARA_PATH/${pair}
    unzip -u $PARA_PATH/${pair}/download.php?f=OpenSubtitles2018%2Fes-it.txt.zip -d $PARA_PATH/${pair}
  fi
else
  echo "source error : $SRC is not valid source, choose between MultiUN and OpenSubtitles2018"
  exit
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

# tokenize
echo -e "\n"
echo "*** Cleaning and tokenizing $pair data ... ***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  if [ ! -f $PARA_PATH/${pair}/$pair.$lg.all ]; then
    if [ $N_SAMPLES = "false" ];then
      cat $PARA_PATH/${pair}/*.$pair.$lg | $TOKENIZE $lg $threads_for_tokenizer | python $LOWER_REMOVE_ACCENT > $PARA_PATH/${pair}/$pair.$lg.all
      #cp $PARA_PATH/${pair}/$SRC.$pair.$lg $PARA_PATH/${pair}/$pair.$lg.all
    else
      cat $PARA_PATH/${pair}/*.$pair.$lg > $PARA_PATH/${pair}/all.$pair.$lg
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
done

echo -e "\n\n"
echo "***build the training set for BPE tokenization ($nCodes codes)***"

echo -e "\n"
echo "***shuf ... Generating $shuf_n_samples random permutations of train data and store result in $OUTPATH/${pair}/bpe.train***"
if [ ! -f $OUTPATH/${pair}/bpe.train ]; then
  for lg in $(echo $pair | sed -e 's/\-/ /g'); do
    shuf -r -n $shuf_n_samples $PARA_PATH/${pair}/$pair.$lg.train >> $OUTPATH/${pair}/bpe.train
  done
else
  #rm $OUTPATH/${pair}/bpe.train
  echo "file $OUTPATH/${pair}/bpe.train already exists"
fi

echo -e "\n"
echo "***Learn the BPE vocabulary on the training set : $OUTPATH/bpe.train***"
if [ ! -f $OUTPATH/${pair}/codes ]; then
  $FASTBPE learnbpe $nCodes $OUTPATH/${pair}/bpe.train > $OUTPATH/${pair}/codes
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
  done
done

echo -e "\n"
echo "***Using parallel data to construct monolingual data***"
for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    cp $OUTPATH/${pair}/$pair.$lg.$split.pth $OUTPATH/${pair}/$split.$lg.pth
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