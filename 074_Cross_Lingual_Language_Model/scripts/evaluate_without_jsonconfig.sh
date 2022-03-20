#!/bin/bash

# required :
#   exp_name
#   exp_id
#   dump_path 
#   data_path
#   lgs
#   mlm_steps
#   epoch_size
#   batch_size
#   max_epoch
#   stopping_criterion
#   eval_bleu
#   remove_long_sentences_train  
#   remove_long_sentences_valid  
#   remove_long_sentences_test 
#   train_n_samples 
#   valid_n_samples 
#   test_n_samples

# Usage: .evaluate.sh  

set -e

python train.py --eval_only True --exp_name $exp_name --exp_id $exp_id --dump_path $dump_path --data_path $data_path --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples
