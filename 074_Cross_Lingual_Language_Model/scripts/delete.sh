#!/bin/bash

# required :
#   src_path 
#   tgt_path
#   $src_pair 
#   $tgt_pair

# Usage: ./duplicate.sh $src_path $tgt_path $src_pair $tgt_pair

set -e

echo "====================="

if [ ! -d $tgt_path ]; then
    mkdir $tgt_path
fi

# rename path tgt_path lang1-lang2
delete() {
  
    IFS='- ' read -r -a langs <<< "$2"
    for (( i=0; i<${#langs[*]}; ++i)); do 
        lang=${langs[$i]}
        for split in valid test; do
            echo "delete $1/$split.$lang.pth"
            rm $1/$split.$lang.pth
        done
    done
}

delete $1 $2 

echo "====================="