#!/bin/bash

# required :
#   src_lan in en,de,ru 
#   tgt_path
#   $src_pair 
#   $tgt_pair

# Usage: ./duplicate.sh $src_path $tgt_path $src_pair $tgt_pair

set -e

if src_lan not in en,de,ru  :
    exit

# download and extract the model
! wget -c https://dl.fbaipublicfiles.com/fairseq/models/lm/wmt19.en.tar.gz -P /home/jupyter/test
! tar -xvzf /home/jupyter/test/wmt19.en.tar.gz -C /home/jupyter/test