## Note : === is Markdown

[1]
%bookmark HOME "/home/jupyter/meta_XLM/XLM" 
%cd -b HOME

########### Prepare txt
[2]
%env csv_path=/home/jupyter
%env output_dir=/home/jupyter/data/evaluation_hypothesis
%env data_type=mono

[3]
%env languages=Bafia,Bafia
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

%env languages=Bulu,Bulu
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

%env languages=MKPAMAN_AMVOE_Ewondo,MKPAMAN_AMVOE_Ewondo
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

%env languages=Ghomala,Ghomala
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

%env languages=Limbum,Limbum
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

%env languages=Ngiemboon,Ngiemboon
! python ../bible.py --csv_path $csv_path --output_dir $output_dir --data_type $data_type --languages $languages

########### Prepare pth

[4]
%env src_path=/home/jupyter/data/evaluation_hypothesis
%env threads_for_tokenizer=16 
%env n_samples=-1
%env test_size=10             
%env val_size=10
# tools paths
%env TOKENIZE=tools/tokenizer_our.sh
%env LOWER_REMOVE_ACCENT=tools/lowercase_and_remove_accent.py
%env FASTBPE=tools/fastBPE/fast
! chmod +x $FASTBPE
! chmod +x tools/mosesdecoder/scripts/tokenizer/*.perl
! chmod +x ../build_evaluate_data.sh

[5] Bafia_Bulu

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Bafia_Bulu
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster1/data/Bafia_Bulu/processed
./build_evaluate_data.sh MKPAMAN_AMVOE_Ewondo $n_samples
./build_evaluate_data.sh Ghomala $n_samples
./build_evaluate_data.sh Limbum $n_samples
./build_evaluate_data.sh Ngiemboon $n_samples

[6] Bafia_Ewondo

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Bafia_Ewondo
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster1/data/Bafia_Ewondo/processed
./build_evaluate_data.sh Bulu $n_samples
./build_evaluate_data.sh Ghomala $n_samples
./build_evaluate_data.sh Limbum $n_samples
./build_evaluate_data.sh Ngiemboon $n_samples

[7] Bulu_Ewondo

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Bulu_Ewondo
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster1/data/Bulu_MKPAMAN_AMVOE_Ewondo/processed
./build_evaluate_data.sh Bafia $n_samples
./build_evaluate_data.sh Ghomala $n_samples
./build_evaluate_data.sh Limbum $n_samples
./build_evaluate_data.sh Ngiemboon $n_samples

[8] Ghomala_Limbum

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Ghomala_Limbum
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster3/data/Ghomala_Limbum/processed
./build_evaluate_data.sh Bafia $n_samples
./build_evaluate_data.sh Bulu $n_samples
./build_evaluate_data.sh MKPAMAN_AMVOE_Ewondo $n_samples
./build_evaluate_data.sh Ngiemboon $n_samples

[9] Ghomala_Ngiemboon

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Ghomala_Ngiemboon
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster3/data/Ghomala_Ngiemboon/processed
./build_evaluate_data.sh Bafia $n_samples
./build_evaluate_data.sh Bulu $n_samples
./build_evaluate_data.sh MKPAMAN_AMVOE_Ewondo $n_samples
./build_evaluate_data.sh Limbum $n_samples

[10] Limbum_Ngiemboon

%env tgt_path=/home/jupyter/models/africa/evaluation_hypothesis/Limbum_Ngiemboon
%env CODE_VOCAB_PATH=/home/jupyter/models/africa/cluster3/data/Limbum_Ngiemboon/processed
./build_evaluate_data.sh Bafia $n_samples
./build_evaluate_data.sh Bulu $n_samples
./build_evaluate_data.sh MKPAMAN_AMVOE_Ewondo $n_samples
./build_evaluate_data.sh Ghomala $n_samples

## Start evaluation

[12]
%env exp_id=maml
%env batch_size=32
%env max_epoch=100
%env stopping_criterion=_valid_mlm_ppl,10
%env eval_bleu=False
%env remove_long_sentences_train=True
%env remove_long_sentences_valid=True
%env remove_long_sentences_test=True
%env train_n_samples=-1
%env valid_n_samples=-1
%env test_n_samples=-1


[13] Bafia_Bulu

%env epoch_size=
%env dump_path=
%env exp_name=
%env data_path=/home/jupyter/models/africa/avaluation_hypothesis/Bafia_Bulu
%env lgs=Bafia-Bulu
%env mlm_steps=Bafia,Bulu,Bafia-Bulu
%env tgt_pair=Bafia-Bulu
%env src_path=/home/jupyter/models/africa/avaluation_hypothesis/Bafia_Bulu
%env tgt_path=/home/jupyter/models/africa/avaluation_hypothesis/Bafia_Bulu

[14]======# vs Ewondo and Ghomala
../copy_rename.sh $src_path $tgt_path Ewondo-Ghomala $tgt_pair
../evaluate.sh

[15]======# vs Limbum and Ngiemboon
../copy_rename.sh $src_path $tgt_path Ewondo-Ghomala $tgt_pair
../evaluate.sh
