python run_language_modeling.py \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --line_by_line \
    --per_device_train_batch_size 16 \
    --train_data_file $1~/eshaan/xword/segmenter/data/processed/train.txt \
    --output_dir ~/eshaan/xword/checkpoints/segmenter/ \
    --save_steps 25000 \
    --do_train

