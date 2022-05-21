# Install Dependencies
pip install scipy
mv DPR/transformers3 ./
pip install -e DPR/
mv transformers3 DPR/
pip install -e DPR/transformers3/
pip install -e DrQA/
pip install transformers
pip install wordsegment
pip install git+https://github.com/alexdej/puzpy.git

# Download checkpoints
mkdir -p checkpoints/byt5_reranker
mkdir -p checkpoints/biencoder/embeddings
mkdir -p checkpoints/gpt2_segmenter
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/biencoder/dpr_biencoder.bin -O checkpoints/biencoder/dpr_biencoder.bin
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/biencoder/wordlist.tsv -O checkpoints/biencoder/wordlist.tsv
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/byt5_reranker/pytorch_model.bin -O checkpoints/byt5_reranker/pytorch_model.bin
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/byt5_reranker/config.json -O checkpoints/byt5_reranker/config.json
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/byt5_reranker/special_tokens_map.json -O checkpoints/byt5_reranker/special_tokens_map.json
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/byt5_reranker/tokenizer_config.json -O checkpoints/byt5_reranker/tokenizer_config.json
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/segmenter/pytorch_model.bin -O checkpoints/gpt2_segmenter/pytorch_model.bin
wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/segmenter/config.json -O checkpoints/gpt2_segmenter/config.json
# Download precomputed embeddings
for i in {0..3}; do
    wget https://huggingface.co/albertxu/Berkeley-Crossword-Solver/resolve/main/biencoder/embeddings/embeddings.json_$i.pkl -O checkpoints/biencoder/embeddings/embeddings.json_$i.pkl
done
