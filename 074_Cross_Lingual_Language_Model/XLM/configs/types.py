config_dic = {
    # main parameters
    "dump_path":[str, "./dumped/"],
    "exp_name":[str, ""],
    "save_periodic":[int, 0],
    "exp_id":[str, ""],
    # AMP API
    "fp16":[bool, False],
    "amp":[int, -1],
    # only use an encoder (use a specific decoder for machine translation)
    "encoder_only":[bool, True],
    # model parameters
    "emb_dim":[int, 512], 
    "n_layers":[int, 4],
    "n_heads":[int, 8],
    "dropout":[float, 0],
    "attention_dropout":[float, 0],
    "gelu_activation":[bool, False], 
    "share_inout_emb":[bool, True], 
    "sinusoidal_embeddings":[bool, False],
    "use_lang_emb":[bool, True],
    # memory parameters
    "use_memory":[bool, False],
    "mem_enc_positions":[str, ""],
    "mem_dec_positions":[str, ""],
    # adaptive softmax
    "asm":[bool, False],
    "asm_cutoffs":[str, "8000,20000"],
    "asm_div_value":[float, 4],
    # causal language modeling task parameters
    "context_size":[int, 0],
    # masked language modeling task parameters
    "word_pred":[float, 0.15],
    "sample_alpha":[float, 0],
    "word_mask_keep_rand":[str, "0.8,0.1,0.1"],
    # input sentence noise
    "word_shuffle":[float, 0], 
    "word_dropout":[float, 0],
    "word_blank":[float, 0],
    # data
    "data_path":[str, ""],
    "lgs":[str, ""],
    "max_vocab":[int, -1],
    "min_count":[int, 0],
    "lg_sampling_factor":[float, -1],
    # batch parameters
    "bptt":[int, 256],
    "max_len":[int, 100],
    "group_by_size":[bool, True],
    "batch_size":[int, 32],
    "max_batch_size":[int, 0],
    "tokens_per_batch":[int, -1],
    # training parameters
    "split_data":[bool, False],
    "optimizer":[str, "adam,lr=0.0001"],
    "clip_grad_norm":[float, 5],
    "epoch_size":[int, 100000],
    "max_epoch":[int, 100000],
    "stopping_criterion":[str, ""],
    "validation_metrics":[str, ""],
    "accumulate_gradients":[int, 1],
    # training coefficients
    "lambda_mlm":[str, "1"],
    "lambda_clm":[str, "1"],
    "lambda_pc":[str, "1"], 
    "lambda_ae":[str, "1"],
    "lambda_mt":[str, "1"],
    "lambda_bt":[str, "1"],
    # training steps
    "clm_steps":[str, ""],
    "mlm_steps":[str, ""], 
    "mt_steps":[str, ""],
    "ae_steps":[str, ""], 
    "bt_steps":[str, ""], 
    "pc_steps":[str, ""],
    # reload pretrained embeddings / pretrained model / checkpoint, 1],
    "reload_emb":[str, ""],
    "reload_model":[str, ""],
    "reload_checkpoint":[str, ""],
    # beam search (for MT only)
    "beam_size":[int, 1],
    "length_penalty":[float, 1],
    "early_stopping":[bool, False],
    # evaluation
    "eval_bleu":[bool, False],
    "eval_only":[bool, False],
    # debug
    "debug_train":[bool, False],
    "debug_slurm":[bool, False],
    ###"debug":?
    # multi-gpu / multi-node
    "local_rank":[int, -1],
    "master_port":[int, -1],
    # our
    "train_n_samples":[int, -1],
    "valid_n_samples":[int, -1],
    "test_n_samples":[int, -1],
    "remove_long_sentences_train":[bool, False],
    "remove_long_sentences_valid":[bool, False],
    "remove_long_sentences_test":[bool, False],
    "same_data_path":[bool, True],
    #"config_file":[str, ""],
    #"log_file_prefix":[str, ""],
    "aggregation_metrics":[str, ""],
    "eval_tasks":[str, ""],
}