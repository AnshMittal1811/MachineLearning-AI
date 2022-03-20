#!/bin/bash

# required :
#   src_path 
#   tgt_path
#   $src_pair 
#   $tgt_pair

# Usage: ./copy_rename.sh $src_path $tgt_path $src_pair $tgt_pair

set -e

echo "====================="

if [ ! -d $tgt_path ]; then
    mkdir $tgt_path
fi

# rename src_path tgt_path src_lang1-src_lang2 tgt_lang1-tgt_lang2
rename() {
  
    IFS='- ' read -r -a src_langs <<< "$3"
    IFS='- ' read -r -a tgt_langs <<< "$4"
    for (( i=0; i<${#tgt_langs[*]}; ++i)); do 
        tgt_lang=${tgt_langs[$i]}
        src_lang=${src_langs[$i]}
        echo "$src_lang to $tgt_lang"
        for split in valid test; do
            if [ $1 != $2 ]; then
                # copy
                cp $1/$src_lang/$split.$tgt_lang.pth $2
            fi
            # rename
            mv $2/$split.$src_lang.pth $2/$split.$tgt_lang.pth
        done
    done
}

rename $1 $2 $3 $4

echo "====================="