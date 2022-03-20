#!/bin/bash

# required :
#   src_file 
#   tgt_file
#   n_samples 


# Usage: ./samples.sh $src_file $tgt_file $n_samples

set -e

src_file=$1
tgt_file=$2
N_SAMPLES=${3-'False'}
if [ $N_SAMPLES -le 0 ];then
    N_SAMPLES=False
fi

echo "==========================="
if [ $N_SAMPLES = "False" ];then
    mv $src_file $tgt_file
    echo "move $src_file to $tgt_file "
else
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
    
    get_n_samples $src_file $N_SAMPLES $tgt_file
    echo "select $N_SAMPLES of $src_file to $tgt_file "
fi
echo "==========================="
