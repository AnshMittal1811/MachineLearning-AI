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

[3]======# XLM Ghomala Limbum

[4]
%env OUTPATH=/home/jupyter/models/africa/cluster3/data/Ghomala_Limbum/processed
%env epoch_size=6333
%env lgs=Ghomala-Limbum
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster3

[5]============#### MLM + TLM

[6]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Ghomala,Limbum,Ghomala-Limbum
! python train.py --eval_only $eval_only --exp_name mlm_tlm_GhomalaLimbum --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[7]============#### MT

[8]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Ghomala-Limbum_mt_bleu,10
%env validation_metrics=valid_Ghomala-Limbum_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaLimbum/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaLimbum/maml/best-valid_mlm_ppl.pth
%env ae_steps=Ghomala,Limbum
%env bt_steps=Ghomala-Limbum-Ghomala,Limbum-Ghomala-Limbum
%env mt_steps=Ghomala-Limbum,Limbum-Ghomala  

%env mlm_steps=Ghomala,Limbum,Ghomala-Limbum     
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_GhomalaLimbum --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test   


[9]======# XLM Ghomala Ngiemboon

[10]
%env OUTPATH=/home/jupyter/models/africa/cluster3/data/Ghomala_Ngiemboon/processed
%env epoch_size=6339
%env lgs=Ghomala-Ngiemboon
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster3

[11]============#### MLM + TLM

[12]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Ghomala,Ngiemboon,Ghomala-Ngiemboon
! python train.py --eval_only $eval_only --exp_name mlm_tlm_GhomalaNgiemboon --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[13]============#### MT

[14]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Ghomala-Ngiemboon_mt_bleu,10
%env validation_metrics=valid_Ghomala-Ngiemboon_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaNgiemboon/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaNgiemboon/maml/best-valid_mlm_ppl.pth
%env ae_steps=Ghomala,Ngiemboon
%env bt_steps=Ghomala-Ngiemboon-Ghomala,Ngiemboon-Ghomala-Ngiemboon
%env mt_steps=Ghomala-Ngiemboon,Ngiemboon-Ghomala       

%env mlm_steps=Ghomala,Ngiemboon,Ghomala-Ngiemboon
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_GhomalaNgiemboon --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

[15]======# XLM Limbum Ngiemboon

[16]
%env OUTPATH=/home/jupyter/models/africa/cluster3/data/Limbum_Ngiemboon/processed
%env epoch_size=6322
%env lgs=Limbum-Ngiemboon
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster3

[17]============#### MLM + TLM

[18]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Limbum,Ngiemboon,Limbum-Ngiemboon
! python train.py --eval_only $eval_only --exp_name mlm_tlm_LimbumNgiemboon --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[19]============#### MT

[20]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Limbum-Ngiemboon_mt_bleu,10
%env validation_metrics=valid_Limbum-Ngiemboon_mt_bleu
%env reload_model=/home/jupyter/models/africa/cluster3/mlm_tlm_LimbumNgiemboon/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster3/mlm_tlm_LimbumNgiemboon/maml/best-valid_mlm_ppl.pth
%env ae_steps=Limbum,Ngiemboon
%env bt_steps=Limbum-Ngiemboon-Limbum,Ngiemboon-Limbum-Ngiemboon
%env mt_steps=Limbum-Ngiemboon,Ngiemboon-Limbum          

%env mlm_steps=Limbum,Ngiemboon,Limbum-Ngiemboon
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_LimbumNgiemboon --exp_id $exp_id  --dump_path ./dumped/ --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    


[21]======# XLM Ghomala Limbum Ngiemboon = XLM_cluster1

[22]
%env OUTPATH=/home/jupyter/models/africa/cluster3/data/XLM_all/processed
%env epoch_size=6339
%env lgs=Ghomala-Limbum-Ngiemboon
%env batch_size=32
%env max_epoch=100
%env dump_path=/home/jupyter/models/africa/cluster3

[23]============#### MLM + TLM

[24]
# stopping criterion (if criterion does not improve 10 times)
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env mlm_steps=Ghomala,Limbum,Ngiemboon,Ghomala-Limbum,Ghomala-Ngiemboon,Limbum-Ngiemboon
! python train.py --eval_only $eval_only --exp_name mlm_tlm_GhomalaLimbumNgiemboon --exp_id $exp_id --dump_path $dump_path --data_path $OUTPATH --lgs $lgs --clm_steps '' --mlm_steps $mlm_steps --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --batch_size $batch_size --bptt 256 --optimizer adam,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --validation_metrics _valid_mlm_ppl --stopping_criterion $stopping_criterion --eval_bleu $eval_bleu --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples

[25]============#### MT

[26]
%env eval_bleu=True
! chmod +x src/evaluation/multi-bleu.perl

%env stopping_criterion=valid_Ghomala-Limbum_mt_bleu,10
%env validation_metrics=valid_Ghomala-Limbum_mt_bleu 
%env reload_model=/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaLimbumNgiemboon/maml/best-valid_mlm_ppl.pth,/home/jupyter/models/africa/cluster3/mlm_tlm_GhomalaLimbumNgiemboon/maml/best-valid_mlm_ppl.pth
%env ae_steps=Ghomala,Limbum,Ngiemboon
%env bt_steps=Ghomala-Limbum-Ghomala,Limbum-Ghomala-Limbum,Limbum-Ngiemboon-Limbum,Ngiemboon-Limbum-Ngiemboon,Ghomala-Ngiemboon-Ghomala,Ngiemboon-Ghomala-Ngiemboon
%env mt_steps=Ghomala-Limbum,Limbum-Ghomala,Limbum-Ngiemboon,Ngiemboon-Limbum,Ghomala-Ngiemboon,Ngiemboon-Ghomala           

%env mlm_steps=Ghomala,Limbum,Ngiemboon,Ghomala-Limbum,Ghomala-Ngiemboon,Limbum-Ngiemboon
! python train.py --eval_only $eval_only --mlm_steps $mlm_steps --exp_name SupMT_XLM_all --exp_id $exp_id  --dump_path $dump_path --reload_model $reload_model --data_path $OUTPATH --lgs $lgs --ae_steps $ae_steps --mt_steps $mt_steps --bt_steps $bt_steps --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --lambda_ae '0:1,100000:0.1,300000:0' --encoder_only false --emb_dim 1024 --n_layers 6 --n_heads 8 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --tokens_per_batch 2000 --batch_size $batch_size --bptt 256 --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 --epoch_size $epoch_size --max_epoch $max_epoch --eval_bleu $eval_bleu --stopping_criterion $stopping_criterion --validation_metrics $validation_metrics --train_n_samples $train_n_samples --valid_n_samples $valid_n_samples --test_n_samples $test_n_samples --remove_long_sentences_train $remove_long_sentences_train --remove_long_sentences_valid $remove_long_sentences_valid --remove_long_sentences_test $remove_long_sentences_test    

