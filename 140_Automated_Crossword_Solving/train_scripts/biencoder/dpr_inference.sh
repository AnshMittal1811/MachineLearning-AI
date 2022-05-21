MODEL_FILE="./checkpoints/closedbook-bertlarge/dpr_biencoder.19.25777"
ctx_file="./data/closedbook/wordlist.txt"
qa_file="data/closedbook/valid/nyt_mon_wed_2020_2021.tsv"
encoded_ctx_file="data/closedbook-bertlarge/embeddings/embeddings.json_*.pkl"
out_file="data/closedbook-bertlarge/out/retrieved.json"

python3 DPR/dense_retriever.py \
	--model_file ${MODEL_FILE} \
	--ctx_file ${ctx_file} \
	--qa_file ${qa_file} \
	--out_file ${out_file} \
	--n-docs 200 \
	--encoded_ctx_file "${encoded_ctx_file}"
