export DEBUG=" "
export USE_SLURM=0

# 3 view inputs evaluation.
python eval.py --model_path "/path/to/your/model.pth" \
--output_path "/saved/results/path" --name "expname" --src_list "22 25 28" --input_view 3

# 6 view inputs evaluation.
python eval.py --model_path "/path/to/your/model.pth" \
--output_path "/saved/results/path" --name "expname" --src_list "22 25 28 40 44 48" --input_view 6

# 9 view inputs evaluation.
python eval.py --model_path "/path/to/your/model.pth" \
--output_path "/saved/results/path" --name "expname" --src_list "22 25 28 40 44 48 0 8 13" --input_view 9