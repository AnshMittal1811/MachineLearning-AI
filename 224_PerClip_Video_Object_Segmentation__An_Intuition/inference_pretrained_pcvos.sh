GPU=0

EXP_ID="s03_pcvos_ytvos"
MODEL_PATH="saves/s03_pcvos_ytvos.pth"
ICR_OPTION="--refine_clip ICR --T_window 2 --S_window 7 --shared_proj"
PMM_OPTION="--memory_read PMM --predict_all"
MODEL_OPTION="${ICR_OPTION} ${PMM_OPTION}"

################################################################ Youtube VOS 2019 val ################################################################
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2019/${EXP_ID}_c5/ --clip_length 5 $MODEL_OPTION --time
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2019/${EXP_ID}_c10/ --clip_length 10 $MODEL_OPTION --time
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2019/${EXP_ID}_c15/ --clip_length 15 $MODEL_OPTION --time
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2019/${EXP_ID}_c25/ --clip_length 25 $MODEL_OPTION --time

################################################################ Youtube VOS 2018 val ################################################################
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2018/${EXP_ID}_c5/ --clip_length 5 $MODEL_OPTION --time --yv_path ./data/YouTube2018
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2018/${EXP_ID}_c10/ --clip_length 10 $MODEL_OPTION --time --yv_path ./data/YouTube2018
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2018/${EXP_ID}_c15/ --clip_length 15 $MODEL_OPTION --time --yv_path ./data/YouTube2018
CUDA_VISIBLE_DEVICES=$GPU python eval_youtube.py --model $MODEL_PATH --output ./youtube_val_2018/${EXP_ID}_c25/ --clip_length 25 $MODEL_OPTION --time --yv_path ./data/YouTube2018