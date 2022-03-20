#!/bin/bash

# usage : eval_data.sh $languages

set -e

n_samples=-1 

# dossier contenant les données sur la langue à évaluer (doit contenir un fichier nommé $lang.txt)
src_path= 
#dossier où les résultats seront stockés
tgt_path= 
#  dossier conténant le fichier code et vocab issues du pré-traitement de la langue qui évalue
CODE_VOCAB_PATH=

threads_for_tokenizer=16
test_size=100 
val_size=0

chmod +x ../scripts/build_evaluate_data.sh

# langues qu'on évalue
languages=$1
for lang in $(echo $languages | sed -e 's/\,/ /g'); do
    . ../scripts/build_evaluate_data.sh $lang $n_samples
done
