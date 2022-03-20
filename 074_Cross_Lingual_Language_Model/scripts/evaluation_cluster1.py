## Note : === is Markdown

[1]
%env eval_only=True
%env exp_id=maml

[2]
%bookmark HOME "/home/jupyter/meta_XLM/XLM" 
%cd -b HOME

[3]
# If you don't have enough RAM or swap memory, leave these three parameters to True, otherwise you may get an error like this when evaluating 
# RuntimeError: copy_if failed to synchronize: cudaErrorAssert: device-side assert triggered
%env remove_long_sentences_train=True
%env remove_long_sentences_valid=True
%env remove_long_sentences_test=True
#--remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test

# limit the number of examples (-1 by default for non limitation)
%env train_n_samples=-1
%env valid_n_samples=-1
%env test_n_samples=-1
#--train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[4]======# xlm_cluster1_enfr : 

[5]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/multi_xlm_enfr/processed
%env epoch_size=23118
%env lgs=en-fr-Bafia-Bulu-MKPAMAN_AMVOE_Ewondo
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[6]============#### MLM+TLM

[7]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=en,fr,Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-en,Bulu-en,MKPAMAN_AMVOE_Ewondo-en,Bafia-fr,Bulu-fr,MKPAMAN_AMVOE_Ewondo-fr
! python train.py --eval_only $eval_only --exp_name mlm_tlm_cluster1enfr --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[8]============#### MT en cluster1

[9]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl
%env stopping_criterion=valid_en-Bulu_mt_bleu,10
%env validation_metrics=valid_en-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1enfr/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1enfr/maml/best-valid_mlm_ppl.pth
%env ae_steps=en,Bafia,Bulu,MKPAMAN_AMVOE_Ewondo
%env bt_steps=Bafia-en-Bafia,en-Bafia-en,Bulu-en-Bulu,en-Bulu-en,MKPAMAN_AMVOE_Ewondo-en-MKPAMAN_AMVOE_Ewondo,en-MKPAMAN_AMVOE_Ewondo-en
%env mt_steps=Bafia-en,en-Bafia,Bulu-en,en-Bulu,MKPAMAN_AMVOE_Ewondo-en,en-MKPAMAN_AMVOE_Ewondo

%env mlm_steps=en,fr,Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-en,Bulu-en,MKPAMAN_AMVOE_Ewondo-en,Bafia-fr,Bulu-fr,MKPAMAN_AMVOE_Ewondo-fr
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_cluster1en --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[10]============#### MT fr cluster1

[11]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_fr-Bulu_mt_bleu,10
%env validation_metrics=valid_fr-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1enfr/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1enfr/maml/best-valid_mlm_ppl.pth
%env ae_steps=fr,Bafia,Bulu,MKPAMAN_AMVOE_Ewondo
%env bt_steps=Bafia-fr-Bafia,fr-Bafia-fr,Bulu-fr-Bulu,fr-Bulu-fr,MKPAMAN_AMVOE_Ewondo-fr-MKPAMAN_AMVOE_Ewondo,fr-MKPAMAN_AMVOE_Ewondo-fr
%env mt_steps=Bafia-fr,fr-Bafia,Bulu-fr,fr-Bulu,MKPAMAN_AMVOE_Ewondo-fr,fr-MKPAMAN_AMVOE_Ewondo

%env mlm_steps=en,fr,Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-en,Bulu-en,MKPAMAN_AMVOE_Ewondo-en,Bafia-fr,Bulu-fr,MKPAMAN_AMVOE_Ewondo-fr
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_cluster1fr --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[12]======# XLM Bafia Bulu

[13]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/Bafia_Bulu/processed
%env epoch_size=6360
%env lgs=Bafia-Bulu
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[14]============#### MLM + TLM

[15]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bafia,Bulu,Bafia-Bulu
! python train.py --eval_only $eval_only --exp_name mlm_tlm_BafiaBulu --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[16]============#### MT

[17]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bafia-Bulu_mt_bleu,10
%env validation_metrics=valid_Bafia-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaBulu/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaBulu/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bafia,Bulu
%env bt_steps=Bafia-Bulu-Bafia,Bulu-Bafia-Bulu
%env mt_steps=Bafia-Bulu,Bulu-Bafia          

%env mlm_steps=Bafia,Bulu,Bafia-Bulu
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_BafiaBulu --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[18]======# XLM Bafia Ewondo

[19]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/Bafia_Ewondo/processed
%env epoch_size=6358
%env lgs=Bafia-MKPAMAN_AMVOE_Ewondo
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[20]============# MLM+TLM

[21]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bafia,MKPAMAN_AMVOE_Ewondo,Bafia-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --exp_name mlm_tlm_BafiaEwondo --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[22]============#### MT

[23]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bafia-MKPAMAN_AMVOE_Ewondo_mt_bleu,10
%env validation_metrics=valid_Bafia-MKPAMAN_AMVOE_Ewondo_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaEwondo/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaEwondo/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bafia,MKPAMAN_AMVOE_Ewondo
%env bt_steps=Bafia-MKPAMAN_AMVOE_Ewondo-Bafia,MKPAMAN_AMVOE_Ewondo-Bafia-MKPAMAN_AMVOE_Ewondo
%env mt_steps=Bafia-MKPAMAN_AMVOE_Ewondo,MKPAMAN_AMVOE_Ewondo-Bafia 

%env mlm_steps=Bafia,MKPAMAN_AMVOE_Ewondo,Bafia-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_BafiaEwondo --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[24]======# XLM Bulu Ewondo

[25]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/Bulu_MKPAMAN_AMVOE_Ewondo/processed
%env epoch_size=6354
%env lgs=Bulu-MKPAMAN_AMVOE_Ewondo
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[26]============# MLM+TLM

[27]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bulu,MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --exp_name mlm_tlm_BuluMKPAMAN_AMVOE_Ewondo --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[28]============#### MT

[29]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bulu-MKPAMAN_AMVOE_Ewondo_mt_bleu,10
%env validation_metrics=valid_Bulu-MKPAMAN_AMVOE_Ewondo_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_BuluMKPAMAN_AMVOE_Ewondo/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_BuluMKPAMAN_AMVOE_Ewondo/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bulu,MKPAMAN_AMVOE_Ewondo
%env bt_steps=Bulu-MKPAMAN_AMVOE_Ewondo-Bulu,MKPAMAN_AMVOE_Ewondo-Bulu-MKPAMAN_AMVOE_Ewondo
%env mt_steps=Bulu-MKPAMAN_AMVOE_Ewondo,MKPAMAN_AMVOE_Ewondo-Bulu         

%env mlm_steps=Bulu,MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_BuluEwondo --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    


[30]======# XLM Bafia Bulu Ewondo = XLM cluster1

[31]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/XLM_all/processed
%env epoch_size=6360
%env lgs=Bafia-Bulu-MKPAMAN_AMVOE_Ewondo
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[32]============# MLM+TLM

[33]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-Bulu,Bafia-MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --exp_name mlm_tlm_BafiaBuluEwondo --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[34]============#### MT

[35]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bafia-Bulu_mt_bleu,10
%env validation_metrics=valid_Bafia-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaBuluEwondo/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_BafiaBuluEwondo/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bafia,Bulu,MKPAMAN_AMVOE_Ewondo
%env bt_steps=Bafia-Bulu-Bafia,Bulu-Bafia-Bulu,Bafia-MKPAMAN_AMVOE_Ewondo-Bafia,MKPAMAN_AMVOE_Ewondo-Bafia-MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo-Bulu,MKPAMAN_AMVOE_Ewondo-Bulu-MKPAMAN_AMVOE_Ewondo
%env mt_steps=Bafia-Bulu,Bulu-Bafia,Bafia-MKPAMAN_AMVOE_Ewondo,MKPAMAN_AMVOE_Ewondo-Bafia,Bulu-MKPAMAN_AMVOE_Ewondo,MKPAMAN_AMVOE_Ewondo-Bulu     

%env mlm_steps=Bafia,Bulu,MKPAMAN_AMVOE_Ewondo,Bafia-Bulu,Bafia-MKPAMAN_AMVOE_Ewondo,Bulu-MKPAMAN_AMVOE_Ewondo
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_BafiaBuluEwondo --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[30]======# XLM cluster1 cluster3

[31]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/XLM_cluster1_cluster2/processed
%env epoch_size=6360
%env lgs=Bafia-Bulu-Ghomala-Limbum
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[32]============# MLM+TLM

[33]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bafia,Bulu,Ghomala,Limbum,Bafia-Bulu,Ghomala-Limbum,Bulu-Limbum
! python train.py --eval_only $eval_only --exp_name mlm_tlm_cluster1_cluster2 --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[34]============#### MT

[35]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bafia-Bulu_mt_bleu,10
%env validation_metrics=valid_Bafia-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1_cluster2/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1_cluster2/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bafia,Bulu,Ghomala,Limbum
%env bt_steps=Bafia-Bulu-Bafia,Bulu-Bafia-Bulu,Ghomala-Limbum-Ghomala,Limbum-Ghomala-Limbum,Bulu-Limbum-Bulu,Limbum-Bulu-Limbum
%env mt_steps=Bafia-Bulu,Bulu-Bafia,Ghomala-Limbum,Limbum-Ghomala,Bulu-Limbum,Limbum-Bulu

%env mlm_steps=Bafia,Bulu,Ghomala,Limbum,Bafia-Bulu,Ghomala-Limbum,Bulu-Limbum
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_cluster1_cluster2 --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    


[36]======# meta XLM cluster1 cluster3

[37]
%env OUTPATH=/home/jupyter/models/africa/cluster1/data/XLM_cluster1_cluster2/processed
%env epoch_size=6360
%env lgs=Bafia-Bulu-Ghomala-Limbum
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster1

[32]============# MLM+TLM

[33]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Bafia,Bulu,Ghomala,Limbum,Bafia-Bulu,Ghomala-Limbum,Bulu-Limbum
! python train.py --eval_only $eval_only --exp_name mlm_tlm_cluster1_cluster2 --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[34]============#### MT

[35]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Bafia-Bulu_mt_bleu,10
%env validation_metrics=valid_Bafia-Bulu_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1_cluster2/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster1/mlm_tlm_cluster1_cluster2/maml/best-valid_mlm_ppl.pth
%env ae_steps=Bafia,Bulu,Ghomala,Limbum
%env bt_steps=Bafia-Bulu-Bafia,Bulu-Bafia-Bulu,Ghomala-Limbum-Ghomala,Limbum-Ghomala-Limbum,Bulu-Limbum-Bulu,Limbum-Bulu-Limbum
%env mt_steps=Bafia-Bulu,Bulu-Bafia,Ghomala-Limbum,Limbum-Ghomala,Bulu-Limbum,Limbum-Bulu

%env mlm_steps=Bafia,Bulu,Ghomala,Limbum,Bafia-Bulu,Ghomala-Limbum,Bulu-Limbum
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_cluster1_cluster2 --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    
