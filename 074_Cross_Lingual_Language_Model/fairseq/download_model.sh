#!/bin/bash

# required :
#   src_lan (between en, de and ru) 
#   tgt_path


# Usage: download_model.sh $ssrc_lan $tgt_path 

set -e

src_lang=$1

if [ $src_lang != "en" ] && [ $src_lang != "de" ] && [ $src_lang != "ru" ]; then  
    echo "invalid source lang, must be the one of follow : en, de, ru"
    exit
fi

if [ ! -d $tgt_path ]; then
    mkdir $tgt_path
fi

# download and extract the model
wget -c https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.$src_lang.tar.gz -P $tgt_path
tar -xvzf $tgt_path/wmt19.$src_lang.tar.gz -C $tgt_path